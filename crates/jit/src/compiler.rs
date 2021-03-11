//! JIT compilation.

use crate::instantiate::SetupError;
use crate::object::{build_object, ObjectUnwindInfo};
use object::write::Object;
#[cfg(feature = "parallel-compilation")]
use rayon::prelude::*;
use serde_json::json;
use serde::Serialize;
use cranelift_wasm::{GlobalIndex, FuncIndex, WasmType};
use std::hash::{Hash, Hasher};
use std::mem;
use wasmparser::WasmFeatures;
use wasmtime_debug::{emit_dwarf, DwarfSection};
use wasmtime_environ::entity::EntityRef;
use wasmtime_environ::isa::{TargetFrontendConfig, TargetIsa};
use wasmtime_environ::wasm::{DefinedMemoryIndex, MemoryIndex};
use wasmtime_environ::{
    CompiledFunctions, Compiler as EnvCompiler, DebugInfoData, Module, ModuleMemoryOffset,
    ModuleTranslation, Tunables, TypeTables, VMOffsets,
};

/// Select which kind of compilation to use.
#[derive(Copy, Clone, Debug, Hash)]
pub enum CompilationStrategy {
    /// Let Wasmtime pick the strategy.
    Auto,

    /// Compile all functions with Cranelift.
    Cranelift,

    /// Compile all functions with Lightbeam.
    #[cfg(feature = "lightbeam")]
    Lightbeam,
}

/// A WebAssembly code JIT compiler.
///
/// A `Compiler` instance owns the executable memory that it allocates.
///
/// TODO: Evolve this to support streaming rather than requiring a `&[u8]`
/// containing a whole wasm module at once.
///
/// TODO: Consider using cranelift-module.
pub struct Compiler {
    isa: Box<dyn TargetIsa>,
    compiler: Box<dyn EnvCompiler>,
    strategy: CompilationStrategy,
    tunables: Tunables,
    features: WasmFeatures,
}

impl Compiler {
    /// Construct a new `Compiler`.
    pub fn new(
        isa: Box<dyn TargetIsa>,
        strategy: CompilationStrategy,
        tunables: Tunables,
        features: WasmFeatures,
    ) -> Self {
        Self {
            isa,
            strategy,
            compiler: match strategy {
                CompilationStrategy::Auto | CompilationStrategy::Cranelift => {
                    Box::new(wasmtime_cranelift::Cranelift::default())
                }
                #[cfg(feature = "lightbeam")]
                CompilationStrategy::Lightbeam => Box::new(wasmtime_lightbeam::Lightbeam),
            },
            tunables,
            features,
        }
    }
}

fn _assert_compiler_send_sync() {
    fn _assert<T: Send + Sync>() {}
    _assert::<Compiler>();
}

fn transform_dwarf_data(
    isa: &dyn TargetIsa,
    module: &Module,
    debug_data: &DebugInfoData,
    funcs: &CompiledFunctions,
) -> Result<Vec<DwarfSection>, SetupError> {
    let target_config = isa.frontend_config();
    let ofs = VMOffsets::new(target_config.pointer_bytes(), &module);

    let memory_offset = if ofs.num_imported_memories > 0 {
        ModuleMemoryOffset::Imported(ofs.vmctx_vmmemory_import(MemoryIndex::new(0)))
    } else if ofs.num_defined_memories > 0 {
        ModuleMemoryOffset::Defined(ofs.vmctx_vmmemory_definition_base(DefinedMemoryIndex::new(0)))
    } else {
        ModuleMemoryOffset::None
    };
    emit_dwarf(isa, debug_data, funcs, &memory_offset).map_err(SetupError::DebugInfo)
}

#[allow(missing_docs)]
pub struct Compilation {
    pub obj: Object,
    pub unwind_info: Vec<ObjectUnwindInfo>,
    pub funcs: CompiledFunctions,
}

#[allow(missing_docs)]
#[derive(Serialize)]
struct CtWasmMemorySig {
    secret: bool,
    imported: bool,
}

#[allow(missing_docs)]
#[derive(Serialize)]
struct CtWasmFunctionSig {
    trusted: bool,
    #[serde(rename = "paramSecrecy")]
    param_secrecy: Vec<bool>,
    #[serde(rename = "returnSecrecy")]
    return_secrecy: Vec<bool>,
}

impl Compiler {
    /// Return the isa.
    pub fn isa(&self) -> &dyn TargetIsa {
        self.isa.as_ref()
    }

    /// Return the target's frontend configuration settings.
    pub fn frontend_config(&self) -> TargetFrontendConfig {
        self.isa.frontend_config()
    }

    /// Return the tunables in use by this engine.
    pub fn tunables(&self) -> &Tunables {
        &self.tunables
    }

    /// Return the enabled wasm features.
    pub fn features(&self) -> &WasmFeatures {
        &self.features
    }

    fn gen_vmoffsets_info(
        vmoff: &VMOffsets,
        module: &Module,
        types: &TypeTables,
    ) -> Vec<u8> {
        let mut globals: Vec<bool> = Vec::with_capacity(module.globals.len());
        let mut memories: Vec<CtWasmMemorySig> = Vec::with_capacity(module.memory_plans.len());
        let mut functions: Vec<CtWasmFunctionSig> = Vec::with_capacity(module.functions.len());

        for i in 0..module.globals.len() as u32 {
            let index = GlobalIndex::from_u32(i);
            globals.push(match module.globals[index].wasm_ty {
                WasmType::S32 => true,
                WasmType::S64 => true,
                _ => false
            })
        }

        for i in 0..module.memory_plans.len() as u32 {
            let index = MemoryIndex::from_u32(i);
            let memory = &module.memory_plans[index].memory;
            memories.push(CtWasmMemorySig {
                secret: memory.secret,
                imported: module.is_imported_memory(index),
            });
        }

        for i in 0..module.functions.len() as u32 {
            let func_index = FuncIndex::from_u32(i);
            let sig_index = module.functions[func_index];

            let signature = &types.wasm_signatures[sig_index];
            let mut param_secrecy : Vec<bool> = Vec::with_capacity(signature.params.len());
            for param in signature.params.iter() {
                param_secrecy.push(match param {
                    WasmType::S32 => true,
                    WasmType::S64 => true,
                    _ => false
                })
            }
            let mut return_secrecy : Vec<bool> = Vec::with_capacity(signature.returns.len());
            for ret in signature.returns.iter() {
                return_secrecy.push(match ret {
                    WasmType::S32 => true,
                    WasmType::S64 => true,
                    _ => false
                })
            }

            let trusted = signature.trusted;

            functions.push(CtWasmFunctionSig {
                param_secrecy,
                return_secrecy,
                trusted,
            })
        }

        let json = json!({
            "globalsOffset": vmoff.vmctx_globals_begin(),
            "globalsSecrecy": globals,
            "memoriesOffset": vmoff.vmctx_imported_memories_begin(),
            "memories": memories,
            "functions": functions,
        });

        let mut res = json.to_string();
        res.push_str("                    "); //add some extra padding to the json section so we can mess with it for testing later
        
        res.into_bytes()
    }

    /// Compile the given function bodies.
    pub fn compile<'data>(
        &self,
        translation: &mut ModuleTranslation,
        types: &TypeTables,
    ) -> Result<Compilation, SetupError> {
        let functions = mem::take(&mut translation.function_body_inputs);
        let functions = functions.into_iter().collect::<Vec<_>>();
        let funcs = maybe_parallel!(functions.(into_iter | into_par_iter))
            .map(|(index, func)| {
                self.compiler.compile_function(
                    translation,
                    index,
                    func,
                    &*self.isa,
                    &self.tunables,
                    types,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect::<CompiledFunctions>();

        let mut dwarf_sections = if self.tunables.generate_native_debuginfo && !funcs.is_empty() {
            transform_dwarf_data(
                &*self.isa,
                &translation.module,
                &translation.debuginfo,
                &funcs,
            )?
        } else {
            vec![]
        };

        let offsets = VMOffsets::new(self.isa.frontend_config().pointer_bytes(), &translation.module);

        let name = ".vmcontext_offsets";
        let body = Compiler::gen_vmoffsets_info(&offsets, &translation.module, &types);
        let relocs = vec![];

        dwarf_sections.push(DwarfSection {
            name, body, relocs
        });

        let (obj, unwind_info) =
            build_object(&*self.isa, &translation, types, &funcs, dwarf_sections)?;

        Ok(Compilation {
            obj,
            unwind_info,
            funcs,
        })
    }
}

impl Hash for Compiler {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        let Compiler {
            strategy,
            compiler: _,
            isa,
            tunables,
            features,
        } = self;

        // Hash compiler's flags: compilation strategy, isa, frontend config,
        // misc tunables.
        strategy.hash(hasher);
        isa.triple().hash(hasher);
        features.hash(hasher);
        // TODO: if this `to_string()` is too expensive then we should upstream
        // a native hashing ability of flags into cranelift itself, but
        // compilation and/or cache loading is relatively expensive so seems
        // unlikely.
        isa.flags().to_string().hash(hasher);
        isa.frontend_config().hash(hasher);
        tunables.hash(hasher);

        // Catch accidental bugs of reusing across crate versions.
        env!("CARGO_PKG_VERSION").hash(hasher);

        // TODO: ... and should we hash anything else? There's a lot of stuff in
        // `TargetIsa`, like registers/encodings/etc. Should we be hashing that
        // too? It seems like wasmtime doesn't configure it too too much, but
        // this may become an issue at some point.
    }
}
