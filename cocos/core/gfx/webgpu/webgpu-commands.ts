import { CachedArray } from '../../memop/cached-array';
import { error, errorID } from '../../platform';
import { BufferSource, DrawInfo, IndirectBuffer } from '../buffer';
import {
    BufferUsageBit,
    ColorMask,
    CullMode,
    DynamicStateFlagBit,
    Filter,
    Format,
    FormatInfos,
    FormatSize,
    LoadOp,
    MemoryUsageBit,
    SampleCount,
    ShaderStageFlagBit,
    StencilFace,
    TextureFlagBit,
    TextureType,
    Type,
    FormatInfo,
    TextureUsageBit,
    StoreOp,
    ShaderStageFlags,
    DescriptorType,
} from '../define';
import { Color, Rect, Viewport, BufferTextureCopy } from '../define-class';
import { WebGLEXT } from '../webgl/webgl-define';
import { WebGPUCommandAllocator } from './webgpu-command-allocator';
import {
    IWebGPUDepthBias,
    IWebGPUDepthBounds,
    IWebGPUStencilCompareMask,
    IWebGPUStencilWriteMask,
} from './webgpu-command-buffer';
import { WebGPUDevice } from './webgpu-device';
import {
    IWebGPUGPUInputAssembler,
    IWebGPUGPUUniform,
    IWebGPUAttrib,
    IWebGPUGPUDescriptorSet,
    IWebGPUGPUBuffer,
    IWebGPUGPUFramebuffer,
    IWebGPUGPUInput,
    IWebGPUGPUPipelineState,
    IWebGPUGPUSampler,
    IWebGPUGPUShader,
    IWebGPUGPUTexture,
    IWebGPUGPUUniformBlock,
    IWebGPUGPUUniformSampler,
    IWebGPUGPURenderPass,
} from './webgpu-gpu-objects';
import { WebGPURenderPass } from './webgpu-render-pass';
import { RenderPass } from '../render-pass';
import { TextureInfo } from '../texture';
import { linear } from '../../animation/easing';

const WebGPUAdressMode: GPUAddressMode[] = [
    'repeat', // WRAP,
    'mirror-repeat', // MIRROR,
    'clamp-to-edge', // CLAMP,
    'clamp-to-edge', // BORDER,
];

const SAMPLES: number[] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
];

const _f32v4 = new Float32Array(4);

// tslint:disable: max-line-length

function CmpF32NotEuqal (a: number, b: number): boolean {
    const c = a - b;
    return (c > 0.000001 || c < -0.000001);
}

export function GLStageToWebGPUStage (stage: ShaderStageFlags) {
    let flag = 0x0;
    if (stage & ShaderStageFlagBit.VERTEX) { flag |= GPUShaderStage.VERTEX; }
    if (stage & ShaderStageFlagBit.FRAGMENT) { flag |= GPUShaderStage.FRAGMENT; }
    if (stage & ShaderStageFlagBit.COMPUTE) { flag |= GPUShaderStage.COMPUTE; }
    if (stage & ShaderStageFlagBit.ALL) { flag |= (GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE); }
    if (flag === 0x0) { console.error('shader stage not supported by webGPU!'); }
    return flag;
}

export function GLDescTypeToWebGPUDescType (descType: DescriptorType) {
    switch (descType) {
    case DescriptorType.UNIFORM_BUFFER:
        return 'uniform-buffer';
    case DescriptorType.STORAGE_BUFFER:
        return 'storage-buffer';
    case DescriptorType.SAMPLER:
        return 'sampler';
    default:
        console.error('binding type not support by webGPU!');
    }
}

export function GFXFormatToWebGLType (format: Format, gl: WebGL2RenderingContext): GLenum {
    switch (format) {
    case Format.R8: return gl.UNSIGNED_BYTE;
    case Format.R8SN: return gl.BYTE;
    case Format.R8UI: return gl.UNSIGNED_BYTE;
    case Format.R8I: return gl.BYTE;
    case Format.R16F: return gl.HALF_FLOAT;
    case Format.R16UI: return gl.UNSIGNED_SHORT;
    case Format.R16I: return gl.SHORT;
    case Format.R32F: return gl.FLOAT;
    case Format.R32UI: return gl.UNSIGNED_INT;
    case Format.R32I: return gl.INT;

    case Format.RG8: return gl.UNSIGNED_BYTE;
    case Format.RG8SN: return gl.BYTE;
    case Format.RG8UI: return gl.UNSIGNED_BYTE;
    case Format.RG8I: return gl.BYTE;
    case Format.RG16F: return gl.HALF_FLOAT;
    case Format.RG16UI: return gl.UNSIGNED_SHORT;
    case Format.RG16I: return gl.SHORT;
    case Format.RG32F: return gl.FLOAT;
    case Format.RG32UI: return gl.UNSIGNED_INT;
    case Format.RG32I: return gl.INT;

    case Format.RGB8: return gl.UNSIGNED_BYTE;
    case Format.SRGB8: return gl.UNSIGNED_BYTE;
    case Format.RGB8SN: return gl.BYTE;
    case Format.RGB8UI: return gl.UNSIGNED_BYTE;
    case Format.RGB8I: return gl.BYTE;
    case Format.RGB16F: return gl.HALF_FLOAT;
    case Format.RGB16UI: return gl.UNSIGNED_SHORT;
    case Format.RGB16I: return gl.SHORT;
    case Format.RGB32F: return gl.FLOAT;
    case Format.RGB32UI: return gl.UNSIGNED_INT;
    case Format.RGB32I: return gl.INT;

    case Format.BGRA8: return gl.UNSIGNED_BYTE;
    case Format.RGBA8: return gl.UNSIGNED_BYTE;
    case Format.SRGB8_A8: return gl.UNSIGNED_BYTE;
    case Format.RGBA8SN: return gl.BYTE;
    case Format.RGBA8UI: return gl.UNSIGNED_BYTE;
    case Format.RGBA8I: return gl.BYTE;
    case Format.RGBA16F: return gl.HALF_FLOAT;
    case Format.RGBA16UI: return gl.UNSIGNED_SHORT;
    case Format.RGBA16I: return gl.SHORT;
    case Format.RGBA32F: return gl.FLOAT;
    case Format.RGBA32UI: return gl.UNSIGNED_INT;
    case Format.RGBA32I: return gl.INT;

    case Format.R5G6B5: return gl.UNSIGNED_SHORT_5_6_5;
    case Format.R11G11B10F: return gl.UNSIGNED_INT_10F_11F_11F_REV;
    case Format.RGB5A1: return gl.UNSIGNED_SHORT_5_5_5_1;
    case Format.RGBA4: return gl.UNSIGNED_SHORT_4_4_4_4;
    case Format.RGB10A2: return gl.UNSIGNED_INT_2_10_10_10_REV;
    case Format.RGB10A2UI: return gl.UNSIGNED_INT_2_10_10_10_REV;
    case Format.RGB9E5: return gl.FLOAT;

    case Format.D16: return gl.UNSIGNED_SHORT;
    case Format.D16S8: return gl.UNSIGNED_INT_24_8; // no D16S8 support
    case Format.D24: return gl.UNSIGNED_INT;
    case Format.D24S8: return gl.UNSIGNED_INT_24_8;
    case Format.D32F: return gl.FLOAT;
    case Format.D32F_S8: return gl.FLOAT_32_UNSIGNED_INT_24_8_REV;

    case Format.BC1: return gl.UNSIGNED_BYTE;
    case Format.BC1_SRGB: return gl.UNSIGNED_BYTE;
    case Format.BC2: return gl.UNSIGNED_BYTE;
    case Format.BC2_SRGB: return gl.UNSIGNED_BYTE;
    case Format.BC3: return gl.UNSIGNED_BYTE;
    case Format.BC3_SRGB: return gl.UNSIGNED_BYTE;
    case Format.BC4: return gl.UNSIGNED_BYTE;
    case Format.BC4_SNORM: return gl.BYTE;
    case Format.BC5: return gl.UNSIGNED_BYTE;
    case Format.BC5_SNORM: return gl.BYTE;
    case Format.BC6H_SF16: return gl.FLOAT;
    case Format.BC6H_UF16: return gl.FLOAT;
    case Format.BC7: return gl.UNSIGNED_BYTE;
    case Format.BC7_SRGB: return gl.UNSIGNED_BYTE;

    case Format.ETC_RGB8: return gl.UNSIGNED_BYTE;
    case Format.ETC2_RGB8: return gl.UNSIGNED_BYTE;
    case Format.ETC2_SRGB8: return gl.UNSIGNED_BYTE;
    case Format.ETC2_RGB8_A1: return gl.UNSIGNED_BYTE;
    case Format.ETC2_SRGB8_A1: return gl.UNSIGNED_BYTE;
    case Format.ETC2_RGB8: return gl.UNSIGNED_BYTE;
    case Format.ETC2_SRGB8: return gl.UNSIGNED_BYTE;
    case Format.EAC_R11: return gl.UNSIGNED_BYTE;
    case Format.EAC_R11SN: return gl.BYTE;
    case Format.EAC_RG11: return gl.UNSIGNED_BYTE;
    case Format.EAC_RG11SN: return gl.BYTE;

    case Format.PVRTC_RGB2: return gl.UNSIGNED_BYTE;
    case Format.PVRTC_RGBA2: return gl.UNSIGNED_BYTE;
    case Format.PVRTC_RGB4: return gl.UNSIGNED_BYTE;
    case Format.PVRTC_RGBA4: return gl.UNSIGNED_BYTE;
    case Format.PVRTC2_2BPP: return gl.UNSIGNED_BYTE;
    case Format.PVRTC2_4BPP: return gl.UNSIGNED_BYTE;

    case Format.ASTC_RGBA_4x4:
    case Format.ASTC_RGBA_5x4:
    case Format.ASTC_RGBA_5x5:
    case Format.ASTC_RGBA_6x5:
    case Format.ASTC_RGBA_6x6:
    case Format.ASTC_RGBA_8x5:
    case Format.ASTC_RGBA_8x6:
    case Format.ASTC_RGBA_8x8:
    case Format.ASTC_RGBA_10x5:
    case Format.ASTC_RGBA_10x6:
    case Format.ASTC_RGBA_10x8:
    case Format.ASTC_RGBA_10x10:
    case Format.ASTC_RGBA_12x10:
    case Format.ASTC_RGBA_12x12:
    case Format.ASTC_SRGBA_4x4:
    case Format.ASTC_SRGBA_5x4:
    case Format.ASTC_SRGBA_5x5:
    case Format.ASTC_SRGBA_6x5:
    case Format.ASTC_SRGBA_6x6:
    case Format.ASTC_SRGBA_8x5:
    case Format.ASTC_SRGBA_8x6:
    case Format.ASTC_SRGBA_8x8:
    case Format.ASTC_SRGBA_10x5:
    case Format.ASTC_SRGBA_10x6:
    case Format.ASTC_SRGBA_10x8:
    case Format.ASTC_SRGBA_10x10:
    case Format.ASTC_SRGBA_12x10:
    case Format.ASTC_SRGBA_12x12:
        return gl.UNSIGNED_BYTE;

    default: {
        return gl.UNSIGNED_BYTE;
    }
    }
}

export function GFXFormatToWebGLInternalFormat (format: Format): GPUTextureFormat  {
    switch (format) {
    case Format.R8: return 'r8unorm';
    case Format.R8SN: return 'r8snorm';
    case Format.R8UI: return 'r8uint';
    case Format.R8I: return 'r8sint';
    case Format.RG8: return 'rg8unorm';
    case Format.RG8SN: return 'rg8snorm';
    case Format.RG8UI: return 'rg8uint';
    case Format.RG8I: return 'rg8sint';
    case Format.BGRA8: return 'bgra8unorm';
    case Format.RGBA8: return 'rgba8unorm';
    case Format.SRGB8_A8: return 'rgba8unorm-srgb';
    case Format.RGBA8SN: return 'rgba8snorm';
    case Format.RGBA8UI: return 'rgba8uint';
    case Format.RGBA8I: return 'rgba8sint';
    case Format.R16I: return 'r16sint';
    case Format.R16UI: return 'r16uint';
    case Format.R16F: return 'r16sint';
    case Format.RG16I: return 'rg16sint';
    case Format.RG16UI: return 'rg16uint';
    case Format.RG16F: return 'rg16float';
    case Format.RGBA16I: return 'rgba16sint';
    case Format.RGBA16UI: return 'rgba16uint';
    case Format.RGBA16F: return 'rgba16float';
    case Format.R32I: return 'r32sint';
    case Format.R32UI: return 'r32uint';
    case Format.R32F: return 'r32float';
    case Format.RG32I: return 'rg32sint';
    case Format.RG32UI: return 'rg32uint';
    case Format.RG32F: return 'rg32float';
    case Format.RGBA32I: return 'rgba32sint';
    case Format.RGBA32UI: return 'rgba32uint';
    case Format.RGBA32F: return 'rgba32float';
    case Format.RGB10A2: return 'rgb10a2unorm';

    case Format.D24: return 'depth24plus';
    case Format.D32F: return 'depth32float';
    case Format.D32F_S8: return 'depth24plus-stencil8';

    case Format.BC1_ALPHA: return 'bc1-rgba-unorm';
    case Format.BC1_SRGB_ALPHA: return 'bc1-rgba-unorm-srgb';
    case Format.BC2: return 'bc2-rgba-unorm';
    case Format.BC2_SRGB: return 'bc2-rgba-unorm-srgb';
    case Format.BC3: return 'bc3-rgba-unorm';
    case Format.BC3_SRGB: return 'bc3-rgba-unorm-srgb';
    case Format.BC4_SNORM: return 'bc4-r-snorm';
    case Format.BC6H_SF16: return 'bc6h-rgb-sfloat';
    case Format.BC6H_UF16: return 'bc6h-rgb-ufloat';
    case Format.BC7: return 'bc7-rgba-unorm';
    case Format.BC7_SRGB: return 'bc7-rgba-unorm-srgb';

    default: {
        console.error('Unsupported Format, convert to WebGPU internal format failed.');
        return 'rgba8unorm';
    }
    }
}

export function GFXFormatToWebGLFormat (format: Format): GPUTextureFormat {
    return GFXFormatToWebGLInternalFormat(format);
}

export function GFXTextureToWebGPUTexture (textureType: TextureType): GPUTextureViewDimension {
    switch (textureType) {
    case TextureType.TEX1D: return '1d';
    case TextureType.TEX2D: return '2d';
    case TextureType.TEX2D_ARRAY: return '2d-array';
    case TextureType.TEX3D: return '3d';
    case TextureType.CUBE: return 'cube';
    default: {
        console.error('Unsupported textureType, convert to WebGPUTexture failed.');
        return '2d';
    }
    }
}

export function GFXTextureUsageToNative (usage: TextureUsageBit): GPUTextureUsageFlags {
    let nativeUsage: GPUTextureUsageFlags = 0;
    if (usage & TextureUsageBit.TRANSFER_SRC) {
        nativeUsage |= GPUTextureUsage.COPY_SRC;
    }

    if (TextureUsageBit.TRANSFER_DST) {
        nativeUsage |= GPUTextureUsage.COPY_DST;
    }

    if (TextureUsageBit.SAMPLED) {
        nativeUsage |= GPUTextureUsage.SAMPLED;
    }

    if (TextureUsageBit.STORAGE) {
        nativeUsage |= GPUTextureUsage.STORAGE;
    }

    if (TextureUsageBit.COLOR_ATTACHMENT) {
        nativeUsage |= GPUTextureUsage.OUTPUT_ATTACHMENT;
    }

    if (TextureUsageBit.DEPTH_STENCIL_ATTACHMENT) {
        nativeUsage |= GPUTextureUsage.OUTPUT_ATTACHMENT;
    }

    if (typeof nativeUsage === 'undefined') {
        console.error('Unsupported texture usage, convert to webGPU type failed.');
        nativeUsage = GPUTextureUsage.OUTPUT_ATTACHMENT;
    }

    return nativeUsage;
}

function GFXTypeToWebGLType (type: Type, gl: WebGL2RenderingContext): GLenum {
    switch (type) {
    case Type.BOOL: return gl.BOOL;
    case Type.BOOL2: return gl.BOOL_VEC2;
    case Type.BOOL3: return gl.BOOL_VEC3;
    case Type.BOOL4: return gl.BOOL_VEC4;
    case Type.INT: return gl.INT;
    case Type.INT2: return gl.INT_VEC2;
    case Type.INT3: return gl.INT_VEC3;
    case Type.INT4: return gl.INT_VEC4;
    case Type.UINT: return gl.UNSIGNED_INT;
    case Type.FLOAT: return gl.FLOAT;
    case Type.FLOAT2: return gl.FLOAT_VEC2;
    case Type.FLOAT3: return gl.FLOAT_VEC3;
    case Type.FLOAT4: return gl.FLOAT_VEC4;
    case Type.MAT2: return gl.FLOAT_MAT2;
    case Type.MAT2X3: return gl.FLOAT_MAT2x3;
    case Type.MAT2X4: return gl.FLOAT_MAT2x4;
    case Type.MAT3X2: return gl.FLOAT_MAT3x2;
    case Type.MAT3: return gl.FLOAT_MAT3;
    case Type.MAT3X4: return gl.FLOAT_MAT3x4;
    case Type.MAT4X2: return gl.FLOAT_MAT4x2;
    case Type.MAT4X3: return gl.FLOAT_MAT4x3;
    case Type.MAT4: return gl.FLOAT_MAT4;
    case Type.SAMPLER2D: return gl.SAMPLER_2D;
    case Type.SAMPLER2D_ARRAY: return gl.SAMPLER_2D_ARRAY;
    case Type.SAMPLER3D: return gl.SAMPLER_3D;
    case Type.SAMPLER_CUBE: return gl.SAMPLER_CUBE;
    default: {
        console.error('Unsupported GLType, convert to GL type failed.');
        return Type.UNKNOWN;
    }
    }
}

function WebGLTypeToGFXType (glType: GLenum, gl: WebGL2RenderingContext): Type {
    switch (glType) {
    case gl.BOOL: return Type.BOOL;
    case gl.BOOL_VEC2: return Type.BOOL2;
    case gl.BOOL_VEC3: return Type.BOOL3;
    case gl.BOOL_VEC4: return Type.BOOL4;
    case gl.INT: return Type.INT;
    case gl.INT_VEC2: return Type.INT2;
    case gl.INT_VEC3: return Type.INT3;
    case gl.INT_VEC4: return Type.INT4;
    case gl.UNSIGNED_INT: return Type.UINT;
    case gl.UNSIGNED_INT_VEC2: return Type.UINT2;
    case gl.UNSIGNED_INT_VEC3: return Type.UINT3;
    case gl.UNSIGNED_INT_VEC4: return Type.UINT4;
    case gl.UNSIGNED_INT: return Type.UINT;
    case gl.FLOAT: return Type.FLOAT;
    case gl.FLOAT_VEC2: return Type.FLOAT2;
    case gl.FLOAT_VEC3: return Type.FLOAT3;
    case gl.FLOAT_VEC4: return Type.FLOAT4;
    case gl.FLOAT_MAT2: return Type.MAT2;
    case gl.FLOAT_MAT2x3: return Type.MAT2X3;
    case gl.FLOAT_MAT2x4: return Type.MAT2X4;
    case gl.FLOAT_MAT3x2: return Type.MAT3X2;
    case gl.FLOAT_MAT3: return Type.MAT3;
    case gl.FLOAT_MAT3x4: return Type.MAT3X4;
    case gl.FLOAT_MAT4x2: return Type.MAT4X2;
    case gl.FLOAT_MAT4x3: return Type.MAT4X3;
    case gl.FLOAT_MAT4: return Type.MAT4;
    case gl.SAMPLER_2D: return Type.SAMPLER2D;
    case gl.SAMPLER_2D_ARRAY: return Type.SAMPLER2D_ARRAY;
    case gl.SAMPLER_3D: return Type.SAMPLER3D;
    case gl.SAMPLER_CUBE: return Type.SAMPLER_CUBE;
    default: {
        console.error('Unsupported GLType, convert to Type failed.');
        return Type.UNKNOWN;
    }
    }
}

function WebGLGetTypeSize (glType: GLenum, gl: WebGL2RenderingContext): Type {
    switch (glType) {
    case gl.BOOL: return 4;
    case gl.BOOL_VEC2: return 8;
    case gl.BOOL_VEC3: return 12;
    case gl.BOOL_VEC4: return 16;
    case gl.INT: return 4;
    case gl.INT_VEC2: return 8;
    case gl.INT_VEC3: return 12;
    case gl.INT_VEC4: return 16;
    case gl.UNSIGNED_INT: return 4;
    case gl.UNSIGNED_INT_VEC2: return 8;
    case gl.UNSIGNED_INT_VEC3: return 12;
    case gl.UNSIGNED_INT_VEC4: return 16;
    case gl.FLOAT: return 4;
    case gl.FLOAT_VEC2: return 8;
    case gl.FLOAT_VEC3: return 12;
    case gl.FLOAT_VEC4: return 16;
    case gl.FLOAT_MAT2: return 16;
    case gl.FLOAT_MAT2x3: return 24;
    case gl.FLOAT_MAT2x4: return 32;
    case gl.FLOAT_MAT3x2: return 24;
    case gl.FLOAT_MAT3: return 36;
    case gl.FLOAT_MAT3x4: return 48;
    case gl.FLOAT_MAT4x2: return 32;
    case gl.FLOAT_MAT4x3: return 48;
    case gl.FLOAT_MAT4: return 64;
    case gl.SAMPLER_2D: return 4;
    case gl.SAMPLER_2D_ARRAY: return 4;
    case gl.SAMPLER_2D_ARRAY_SHADOW: return 4;
    case gl.SAMPLER_3D: return 4;
    case gl.SAMPLER_CUBE: return 4;
    case gl.INT_SAMPLER_2D: return 4;
    case gl.INT_SAMPLER_2D_ARRAY: return 4;
    case gl.INT_SAMPLER_3D: return 4;
    case gl.INT_SAMPLER_CUBE: return 4;
    case gl.UNSIGNED_INT_SAMPLER_2D: return 4;
    case gl.UNSIGNED_INT_SAMPLER_2D_ARRAY: return 4;
    case gl.UNSIGNED_INT_SAMPLER_3D: return 4;
    case gl.UNSIGNED_INT_SAMPLER_CUBE: return 4;
    default: {
        console.error('Unsupported GLType, get type failed.');
        return 0;
    }
    }
}

function WebGLGetComponentCount (glType: GLenum, gl: WebGL2RenderingContext): Type {
    switch (glType) {
    case gl.FLOAT_MAT2: return 2;
    case gl.FLOAT_MAT2x3: return 2;
    case gl.FLOAT_MAT2x4: return 2;
    case gl.FLOAT_MAT3x2: return 3;
    case gl.FLOAT_MAT3: return 3;
    case gl.FLOAT_MAT3x4: return 3;
    case gl.FLOAT_MAT4x2: return 4;
    case gl.FLOAT_MAT4x3: return 4;
    case gl.FLOAT_MAT4: return 4;
    default: {
        return 1;
    }
    }
}

const WebGLCmpFuncs: GLenum[] = [
    0x0200, // WebGLRenderingContext.NEVER,
    0x0201, // WebGLRenderingContext.LESS,
    0x0202, // WebGLRenderingContext.EQUAL,
    0x0203, // WebGLRenderingContext.LEQUAL,
    0x0204, // WebGLRenderingContext.GREATER,
    0x0205, // WebGLRenderingContext.NOTEQUAL,
    0x0206, // WebGLRenderingContext.GEQUAL,
    0x0207, // WebGLRenderingContext.ALWAYS,
];

const WebGLStencilOps: GLenum[] = [
    0x0000, // WebGLRenderingContext.ZERO,
    0x1E00, // WebGLRenderingContext.KEEP,
    0x1E01, // WebGLRenderingContext.REPLACE,
    0x1E02, // WebGLRenderingContext.INCR,
    0x1E03, // WebGLRenderingContext.DECR,
    0x150A, // WebGLRenderingContext.INVERT,
    0x8507, // WebGLRenderingContext.INCR_WRAP,
    0x8508, // WebGLRenderingContext.DECR_WRAP,
];

const WebGLBlendOps: GLenum[] = [
    0x8006, // WebGLRenderingContext.FUNC_ADD,
    0x800A, // WebGLRenderingContext.FUNC_SUBTRACT,
    0x800B, // WebGLRenderingContext.FUNC_REVERSE_SUBTRACT,
    0x8006, // WebGLRenderingContext.FUNC_ADD,
    0x8006, // WebGLRenderingContext.FUNC_ADD,
];

export const WebGPUStencilOp: GPUStencilOperation[] = [
    'zero',
    'keep',
    'replace',
    'increment-clamp',
    'decrement-clamp',
    'invert',
    'increment-wrap',
    'decrement-wrap',
];

export const WebGPUCompereFunc: GPUCompareFunction[] = [
    'never',
    'less',
    'equal',
    'less-equal',
    'greater',
    'not-equal',
    'greater-equal',
    'always',
];

export const WebGPUBlendOps: GPUBlendOperation[] = [
    'add',
    'subtract',
    'reverse-subtract',
    'min',
    'max',
];

export const WebGPUBlendFactors: GPUBlendFactor[] = [
    'zero',
    'one',
    'src-alpha',
    'dst-alpha',
    'one-minus-src-alpha',
    'one-minus-dst-alpha',
    'src-color',
    'dst-color',
    'one-minus-src-color',
    'one-minus-dst-color',
    'src-alpha-saturated',
    'blend-color', // CONSTANT_COLOR
    'one-minus-blend-color', // ONE_MINUS_CONSTANT_COLOR
    'src-alpha', // CONSTANT_ALPHA: not supported
    'one-minus-src-alpha', // ONE_MINUS_CONSTANT_ALPHA: not supported
];

export enum WebGPUCmd {
    BEGIN_RENDER_PASS,
    END_RENDER_PASS,
    BIND_STATES,
    DRAW,
    UPDATE_BUFFER,
    COPY_BUFFER_TO_TEXTURE,
    COUNT,
}

export abstract class WebGPUCmdObject {
    public cmdType: WebGPUCmd;
    public refCount = 0;

    constructor (type: WebGPUCmd) {
        this.cmdType = type;
    }

    public abstract clear ();
}

export class WebGPUCmdBeginRenderPass extends WebGPUCmdObject {
    public gpuRenderPass: IWebGPUGPURenderPass | null = null;
    public gpuFramebuffer: IWebGPUGPUFramebuffer | null = null;
    public renderArea = new Rect();
    public clearColors: Color[] = [];
    public clearDepth = 1.0;
    public clearStencil = 0;

    constructor () {
        super(WebGPUCmd.BEGIN_RENDER_PASS);
    }

    public clear () {
        this.gpuFramebuffer = null;
        this.clearColors.length = 0;
    }
}

export class WebGPUCmdBindStates extends WebGPUCmdObject {
    public gpuPipelineState: IWebGPUGPUPipelineState | null = null;
    public gpuInputAssembler: IWebGPUGPUInputAssembler | null = null;
    public gpuDescriptorSets: IWebGPUGPUDescriptorSet[] = [];
    public dynamicOffsets: number[] = [];
    public viewport: Viewport | null = null;
    public scissor: Rect | null = null;
    public lineWidth: number | null = null;
    public depthBias: IWebGPUDepthBias | null = null;
    public blendConstants: number[] = [];
    public depthBounds: IWebGPUDepthBounds | null = null;
    public stencilWriteMask: IWebGPUStencilWriteMask | null = null;
    public stencilCompareMask: IWebGPUStencilCompareMask | null = null;

    constructor () {
        super(WebGPUCmd.BIND_STATES);
    }

    public clear () {
        this.gpuPipelineState = null;
        this.gpuInputAssembler = null;
        this.gpuDescriptorSets.length = 0;
        this.dynamicOffsets.length = 0;
        this.viewport = null;
        this.scissor = null;
        this.lineWidth = null;
        this.depthBias = null;
        this.blendConstants.length = 0;
        this.depthBounds = null;
        this.stencilWriteMask = null;
        this.stencilCompareMask = null;
    }
}

export class WebGPUCmdDraw extends WebGPUCmdObject {
    public drawInfo = new DrawInfo();

    constructor () {
        super(WebGPUCmd.DRAW);
    }

    public clear () {
    }
}

export class WebGPUCmdUpdateBuffer extends WebGPUCmdObject {
    public gpuBuffer: IWebGPUGPUBuffer | null = null;
    public buffer: BufferSource | null = null;
    public offset = 0;
    public size = 0;

    constructor () {
        super(WebGPUCmd.UPDATE_BUFFER);
    }

    public clear () {
        this.gpuBuffer = null;
        this.buffer = null;
    }
}

export class WebGPUCmdCopyBufferToTexture extends WebGPUCmdObject {
    public gpuTexture: IWebGPUGPUTexture | null = null;
    public buffers: ArrayBufferView[] = [];
    public regions: BufferTextureCopy[] = [];

    constructor () {
        super(WebGPUCmd.COPY_BUFFER_TO_TEXTURE);
    }

    public clear () {
        this.gpuTexture = null;
        this.buffers.length = 0;
        this.regions.length = 0;
    }
}

export class WebGPUCmdPackage {
    public cmds: CachedArray<WebGPUCmd> = new CachedArray(1);
    public beginRenderPassCmds: CachedArray<WebGPUCmdBeginRenderPass> = new CachedArray(1);
    public bindStatesCmds: CachedArray<WebGPUCmdBindStates> = new CachedArray(1);
    public drawCmds: CachedArray<WebGPUCmdDraw> = new CachedArray(1);
    public updateBufferCmds: CachedArray<WebGPUCmdUpdateBuffer> = new CachedArray(1);
    public copyBufferToTextureCmds: CachedArray<WebGPUCmdCopyBufferToTexture> = new CachedArray(1);

    public clearCmds (allocator: WebGPUCommandAllocator) {
        if (this.beginRenderPassCmds.length) {
            allocator.beginRenderPassCmdPool.freeCmds(this.beginRenderPassCmds);
            this.beginRenderPassCmds.clear();
        }

        if (this.bindStatesCmds.length) {
            allocator.bindStatesCmdPool.freeCmds(this.bindStatesCmds);
            this.bindStatesCmds.clear();
        }

        if (this.drawCmds.length) {
            allocator.drawCmdPool.freeCmds(this.drawCmds);
            this.drawCmds.clear();
        }

        if (this.updateBufferCmds.length) {
            allocator.updateBufferCmdPool.freeCmds(this.updateBufferCmds);
            this.updateBufferCmds.clear();
        }

        if (this.copyBufferToTextureCmds.length) {
            allocator.copyBufferToTextureCmdPool.freeCmds(this.copyBufferToTextureCmds);
            this.copyBufferToTextureCmds.clear();
        }

        this.cmds.clear();
    }
}

export function WebGPUCmdFuncCreateBuffer (device: WebGPUDevice, gpuBuffer: IWebGPUGPUBuffer) {
    const nativeDevice: GPUDevice = device.nativeDevice()!;

    const bufferDesc = {} as GPUBufferDescriptor;
    bufferDesc.size = gpuBuffer.size;

    let bufferUsage = 0x0;

    if (gpuBuffer.usage & BufferUsageBit.VERTEX) bufferUsage |= GPUBufferUsage.VERTEX;
    if (gpuBuffer.usage & BufferUsageBit.INDEX) bufferUsage |= GPUBufferUsage.INDEX;
    if (gpuBuffer.usage & BufferUsageBit.UNIFORM) bufferUsage |= GPUBufferUsage.UNIFORM;
    if (gpuBuffer.usage & BufferUsageBit.INDIRECT) bufferUsage |= GPUBufferUsage.INDIRECT;
    if (gpuBuffer.usage & BufferUsageBit.TRANSFER_SRC) bufferUsage |= GPUBufferUsage.COPY_SRC;
    if (gpuBuffer.usage & BufferUsageBit.TRANSFER_DST) bufferUsage |= GPUBufferUsage.COPY_DST;
    if (gpuBuffer.usage & BufferUsageBit.STORAGE) bufferUsage |= GPUBufferUsage.STORAGE;

    if (bufferUsage === 0x0) {
        console.error('Unsupported GFXBufferType yet, create STORAGE buffer in default.');
        bufferUsage |= GPUBufferUsage.STORAGE;
    }
    bufferDesc.usage = bufferUsage;
    gpuBuffer.glTarget = bufferUsage;
    gpuBuffer.glBuffer = nativeDevice.createBuffer(bufferDesc);

    // const gl = device.gl;
    // const cache = device.stateCache;
    // const glUsage: GLenum = gpuBuffer.memUsage & MemoryUsageBit.HOST ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW;

    // if (gpuBuffer.usage & BufferUsageBit.VERTEX) {

    //     gpuBuffer.glTarget = gl.ARRAY_BUFFER;
    //     const glBuffer = gl.createBuffer();

    //     if (glBuffer) {
    //         gpuBuffer.glBuffer = glBuffer;
    //         if (gpuBuffer.size > 0) {
    //             if (device.useVAO) {
    //                 if (cache.glVAO) {
    //                     gl.bindVertexArray(null);
    //                     cache.glVAO = gfxStateCache.gpuInputAssembler = null;
    //                 }
    //             }

    //             if (device.stateCache.glArrayBuffer !== gpuBuffer.glBuffer) {
    //                 gl.bindBuffer(gl.ARRAY_BUFFER, gpuBuffer.glBuffer);
    //                 device.stateCache.glArrayBuffer = gpuBuffer.glBuffer;
    //             }

    //             gl.bufferData(gl.ARRAY_BUFFER, gpuBuffer.size, glUsage);

    //             gl.bindBuffer(gl.ARRAY_BUFFER, null);
    //             device.stateCache.glArrayBuffer = null;
    //         }
    //     }
    // } else if (gpuBuffer.usage & BufferUsageBit.INDEX) {

    //     gpuBuffer.glTarget = gl.ELEMENT_ARRAY_BUFFER;
    //     const glBuffer = gl.createBuffer();
    //     if (glBuffer) {
    //         gpuBuffer.glBuffer = glBuffer;
    //         if (gpuBuffer.size > 0) {
    //             if (device.useVAO) {
    //                 if (cache.glVAO) {
    //                     gl.bindVertexArray(null);
    //                     cache.glVAO = gfxStateCache.gpuInputAssembler = null;
    //                 }
    //             }

    //             if (device.stateCache.glElementArrayBuffer !== gpuBuffer.glBuffer) {
    //                 gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.glBuffer);
    //                 device.stateCache.glElementArrayBuffer = gpuBuffer.glBuffer;
    //             }

    //             gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.size, glUsage);

    //             gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    //             device.stateCache.glElementArrayBuffer = null;
    //         }
    //     }
    // } else if (gpuBuffer.usage & BufferUsageBit.UNIFORM) {

    //     gpuBuffer.glTarget = gl.UNIFORM_BUFFER;
    //     const glBuffer = gl.createBuffer();
    //     if (glBuffer && gpuBuffer.size > 0) {
    //         gpuBuffer.glBuffer = glBuffer;
    //         if (device.stateCache.glUniformBuffer !== gpuBuffer.glBuffer) {
    //             gl.bindBuffer(gl.UNIFORM_BUFFER, gpuBuffer.glBuffer);
    //             device.stateCache.glUniformBuffer = gpuBuffer.glBuffer;
    //         }

    //         gl.bufferData(gl.UNIFORM_BUFFER, gpuBuffer.size, glUsage);

    //         gl.bindBuffer(gl.UNIFORM_BUFFER, null);
    //         device.stateCache.glUniformBuffer = null;
    //     }
    // } else if (gpuBuffer.usage & BufferUsageBit.INDIRECT) {
    //     gpuBuffer.glTarget = gl.NONE;
    // } else if (gpuBuffer.usage & BufferUsageBit.TRANSFER_DST) {
    //     gpuBuffer.glTarget = gl.NONE;
    // } else if (gpuBuffer.usage & BufferUsageBit.TRANSFER_SRC) {
    //     gpuBuffer.glTarget = gl.NONE;
    // } else {
    //     console.error('Unsupported GFXBufferType, create buffer failed.');
    //     gpuBuffer.glTarget = gl.NONE;
    // }
}

export function WebGPUCmdFuncDestroyBuffer (device: WebGPUDevice, gpuBuffer: IWebGPUGPUBuffer) {
    const gl = device.gl;
    if (gpuBuffer.glBuffer) {
        // Firefox 75+ implicitly unbind whatever buffer there was on the slot sometimes
        // can be reproduced in the static batching scene at https://github.com/cocos-creator/test-cases-3d
        switch (gpuBuffer.glTarget) {
        case gl.ARRAY_BUFFER:
            if (device.useVAO && device.stateCache.glVAO) {
                gl.bindVertexArray(null);
                device.stateCache.glVAO = gfxStateCache.gpuInputAssembler = null;
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, null);
            device.stateCache.glArrayBuffer = null;
            break;
        case gl.ELEMENT_ARRAY_BUFFER:
            if (device.useVAO && device.stateCache.glVAO) {
                gl.bindVertexArray(null);
                device.stateCache.glVAO = gfxStateCache.gpuInputAssembler = null;
            }
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
            device.stateCache.glElementArrayBuffer = null;
            break;
        case gl.UNIFORM_BUFFER:
            gl.bindBuffer(gl.UNIFORM_BUFFER, null);
            device.stateCache.glUniformBuffer = null;
            break;
        }

        gl.deleteBuffer(gpuBuffer.glBuffer);
        gpuBuffer.glBuffer = null;
    }
}

export function WebGPUCmdFuncResizeBuffer (device: WebGPUDevice, gpuBuffer: IWebGPUGPUBuffer) {
    const gl = device.gl;
    const cache = device.stateCache;
    const glUsage: GLenum = gpuBuffer.memUsage & MemoryUsageBit.HOST ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW;

    if (gpuBuffer.usage & BufferUsageBit.VERTEX) {
        if (device.useVAO) {
            if (cache.glVAO) {
                gl.bindVertexArray(null);
                cache.glVAO = gfxStateCache.gpuInputAssembler = null;
            }
        }

        if (cache.glArrayBuffer !== gpuBuffer.glBuffer) {
            gl.bindBuffer(gl.ARRAY_BUFFER, gpuBuffer.glBuffer);
        }

        if (gpuBuffer.buffer) {
            gl.bufferData(gl.ARRAY_BUFFER, gpuBuffer.buffer, glUsage);
        } else {
            gl.bufferData(gl.ARRAY_BUFFER, gpuBuffer.size, glUsage);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        cache.glArrayBuffer = null;
    } else if (gpuBuffer.usage & BufferUsageBit.INDEX) {
        if (device.useVAO) {
            if (cache.glVAO) {
                gl.bindVertexArray(null);
                cache.glVAO = gfxStateCache.gpuInputAssembler = null;
            }
        }

        if (device.stateCache.glElementArrayBuffer !== gpuBuffer.glBuffer) {
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.glBuffer);
        }

        if (gpuBuffer.buffer) {
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.buffer, glUsage);
        } else {
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.size, glUsage);
        }
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
        device.stateCache.glElementArrayBuffer = null;
    } else if (gpuBuffer.usage & BufferUsageBit.UNIFORM) {
        if (device.stateCache.glUniformBuffer !== gpuBuffer.glBuffer) {
            gl.bindBuffer(gl.UNIFORM_BUFFER, gpuBuffer.glBuffer);
        }

        gl.bufferData(gl.UNIFORM_BUFFER, gpuBuffer.size, glUsage);
        gl.bindBuffer(gl.UNIFORM_BUFFER, null);
        device.stateCache.glUniformBuffer = null;
    } else if ((gpuBuffer.usage & BufferUsageBit.INDIRECT)
            || (gpuBuffer.usage & BufferUsageBit.TRANSFER_DST)
            || (gpuBuffer.usage & BufferUsageBit.TRANSFER_SRC)) {
        gpuBuffer.glTarget = gl.NONE;
    } else {
        console.error('Unsupported GFXBufferType, create buffer failed.');
        gpuBuffer.glTarget = gl.NONE;
    }
}

export function WebGPUCmdFuncUpdateBuffer (device: WebGPUDevice, gpuBuffer: IWebGPUGPUBuffer, buffer: BufferSource, offset: number, size: number) {
    if (gpuBuffer.usage & BufferUsageBit.INDIRECT) {
        gpuBuffer.indirects.length = offset;
        Array.prototype.push.apply(gpuBuffer.indirects, (buffer as IndirectBuffer).drawInfos);
    } else {
        const nativeDevice: GPUDevice = device.nativeDevice()!;
        let buff = buffer as ArrayBuffer;
        if (buff.byteLength !== size) {
            buff = buff.slice(0, size);
        }
        // gpuBuffer.glbuffer may not able to be mapped directly, so staging buffer here.
        const stagingBuffer = nativeDevice.createBuffer({
            size,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint8Array(stagingBuffer.getMappedRange(0, size)).set(new Uint8Array(buff));
        stagingBuffer.unmap();

        const cache = device.stateCache;

        const commandEncoder = nativeDevice.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(stagingBuffer, 0, gpuBuffer.glBuffer as GPUBuffer, offset, size);
        const commandBuffer = commandEncoder.finish();
        nativeDevice.defaultQueue.submit([commandBuffer]);
        stagingBuffer.destroy();
        //     switch (gpuBuffer.usage) {
        //         case gl.ARRAY_BUFFER: {
        //             if (cache.glVAO) {
        //                 gl.bindVertexArray(null);
        //                 cache.glVAO = gfxStateCache.gpuInputAssembler = null;
        //             }

        //             if (cache.glArrayBuffer !== gpuBuffer.glBuffer) {
        //                 gl.bindBuffer(gl.ARRAY_BUFFER, gpuBuffer.glBuffer);
        //                 cache.glArrayBuffer = gpuBuffer.glBuffer;
        //             }

        //             if (size === buff.byteLength) {
        //                 gl.bufferSubData(gpuBuffer.glTarget, offset, buff);
        //             } else {
        //                 gl.bufferSubData(gpuBuffer.glTarget, offset, buff.slice(0, size));
        //             }
        //             break;
        //         }
        //         case gl.ELEMENT_ARRAY_BUFFER: {
        //             if (cache.glVAO) {
        //                 gl.bindVertexArray(null);
        //                 cache.glVAO = gfxStateCache.gpuInputAssembler = null;
        //             }

        //             if (cache.glElementArrayBuffer !== gpuBuffer.glBuffer) {
        //                 gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.glBuffer);
        //                 cache.glElementArrayBuffer = gpuBuffer.glBuffer;
        //             }

        //             if (size === buff.byteLength) {
        //                 gl.bufferSubData(gpuBuffer.glTarget, offset, buff);
        //             } else {
        //                 gl.bufferSubData(gpuBuffer.glTarget, offset, buff.slice(0, size));
        //             }
        //             break;
        //         }
        //         case gl.UNIFORM_BUFFER: {
        //             if (cache.glUniformBuffer !== gpuBuffer.glBuffer) {
        //                 gl.bindBuffer(gl.UNIFORM_BUFFER, gpuBuffer.glBuffer);
        //                 cache.glUniformBuffer = gpuBuffer.glBuffer;
        //             }

    //             if (size === buff.byteLength) {
    //                 gl.bufferSubData(gpuBuffer.glTarget, offset, buff);
    //             } else {
    //                 gl.bufferSubData(gpuBuffer.glTarget, offset, new Float32Array(buff, 0, size / 4));
    //             }
    //             break;
    //         }
    //         default: {
    //             console.error('Unsupported GFXBufferType, update buffer failed.');
    //             return;
    //         }
    //     }
    // }
    }
}

export function WebGPUCmdFuncCreateTexture (device: WebGPUDevice, gpuTexture: IWebGPUGPUTexture) {
    // dimension optional
    // let dim: GPUTextureViewDimension = GFXTextureToWebGPUTexture(gpuTexture.type);

    gpuTexture.glTarget = GFXTextureToWebGPUTexture(gpuTexture.type) as GPUTextureDimension;
    gpuTexture.glInternalFmt = GFXFormatToWebGLInternalFormat(gpuTexture.format);
    gpuTexture.glFormat = GFXFormatToWebGLFormat(gpuTexture.format);
    gpuTexture.glUsage = GFXTextureUsageToNative(gpuTexture.usage);
    gpuTexture.glWrapS = gpuTexture.isPowerOf2 ? 'repeat' : 'clamp-to-edge';
    gpuTexture.glWrapT = gpuTexture.isPowerOf2 ? 'repeat' : 'clamp-to-edge';
    gpuTexture.glMinFilter = 'linear';
    gpuTexture.glMagFilter = 'linear';
    // TBD: 2021 feb 2nd only 1 and 4 supported.
    gpuTexture.samples = gpuTexture.samples > 1 ? 4 : 1;
    const texDescriptor: GPUTextureDescriptor = {
        size: [gpuTexture.width, gpuTexture.height, gpuTexture.depth],
        mipLevelCount: gpuTexture.mipLevel,
        sampleCount: gpuTexture.samples,
        format: gpuTexture.glFormat,
        usage: gpuTexture.glUsage,
    };

    gpuTexture.glTexture = device.nativeDevice()?.createTexture(texDescriptor)!;
}

export function WebGPUCmdFuncDestroyTexture (gpuTexture: IWebGPUGPUTexture) {
    if (gpuTexture.glTexture) {
        gpuTexture.glTexture.destroy();
    }
}

export function WebGPUCmdFuncResizeTexture (device: WebGPUDevice, gpuTexture: IWebGPUGPUTexture) {
    if (gpuTexture.glTexture) {
        WebGPUCmdFuncDestroyTexture(gpuTexture);
    }
    WebGPUCmdFuncCreateTexture(device, gpuTexture);
}

export function WebGPUCmdFuncCreateSampler (device: WebGPUDevice, gpuSampler: IWebGPUGPUSampler) {
    const nativeDevice: GPUDevice = device.nativeDevice()!;

    gpuSampler.glMinFilter = (gpuSampler.minFilter === Filter.LINEAR || gpuSampler.minFilter === Filter.ANISOTROPIC) ? 'linear' : 'nearest';
    gpuSampler.glMagFilter = (gpuSampler.magFilter === Filter.LINEAR || gpuSampler.magFilter === Filter.ANISOTROPIC) ? 'linear' : 'nearest';
    gpuSampler.glMipFilter = (gpuSampler.mipFilter === Filter.LINEAR || gpuSampler.mipFilter === Filter.ANISOTROPIC) ? 'linear' : 'nearest';
    gpuSampler.glWrapS = WebGPUAdressMode[gpuSampler.addressU];
    gpuSampler.glWrapT = WebGPUAdressMode[gpuSampler.addressV];
    gpuSampler.glWrapR = WebGPUAdressMode[gpuSampler.addressW];

    const samplerDesc = {} as GPUSamplerDescriptor;
    samplerDesc.addressModeU = gpuSampler.glWrapS;
    samplerDesc.addressModeV = gpuSampler.glWrapT;
    samplerDesc.addressModeW = gpuSampler.glWrapT;
    samplerDesc.minFilter = gpuSampler.glMinFilter;
    samplerDesc.magFilter = gpuSampler.glMagFilter;
    samplerDesc.mipmapFilter = gpuSampler.glMipFilter;
    samplerDesc.lodMinClamp = gpuSampler.minLOD;
    samplerDesc.lodMaxClamp = gpuSampler.maxLOD;

    const sampler: GPUSampler = nativeDevice.createSampler(samplerDesc);
    gpuSampler.glSampler = sampler;
}

export function WebGPUCmdFuncDestroySampler (device: WebGPUDevice, gpuSampler: IWebGPUGPUSampler) {
    if (gpuSampler.glSampler) {
        device.gl.deleteSampler(gpuSampler.glSampler);
        gpuSampler.glSampler = null;
    }
}

export function WebGPUCmdFuncCreateFramebuffer (device: WebGPUDevice, gpuFramebuffer: IWebGPUGPUFramebuffer) {
    if (!gpuFramebuffer.gpuColorTextures.length && !gpuFramebuffer.gpuDepthStencilTexture) { return; } // onscreen fbo

    const gl = device.gl;
    const attachments: GLenum[] = [];

    const glFramebuffer = gl.createFramebuffer();
    if (glFramebuffer) {
        gpuFramebuffer.glFramebuffer = glFramebuffer;

        if (device.stateCache.glFramebuffer !== gpuFramebuffer.glFramebuffer) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, gpuFramebuffer.glFramebuffer);
        }

        for (let i = 0; i < gpuFramebuffer.gpuColorTextures.length; ++i) {
            const colorTexture = gpuFramebuffer.gpuColorTextures[i];
            if (colorTexture) {
                if (colorTexture.glTexture) {
                    gl.framebufferTexture2D(
                        gl.FRAMEBUFFER,
                        gl.COLOR_ATTACHMENT0 + i,
                        colorTexture.glTarget,
                        colorTexture.glTexture,
                        0,
                    ); // level should be 0.
                } else {
                    gl.framebufferRenderbuffer(
                        gl.FRAMEBUFFER,
                        gl.COLOR_ATTACHMENT0 + i,
                        gl.RENDERBUFFER,
                        colorTexture.glRenderbuffer,
                    );
                }

                attachments.push(gl.COLOR_ATTACHMENT0 + i);
            }
        }

        const dst = gpuFramebuffer.gpuDepthStencilTexture;
        if (dst) {
            const glAttachment = FormatInfos[dst.format].hasStencil ? gl.DEPTH_STENCIL_ATTACHMENT : gl.DEPTH_ATTACHMENT;
            if (dst.glTexture) {
                gl.framebufferTexture2D(
                    gl.FRAMEBUFFER,
                    glAttachment,
                    dst.glTarget,
                    dst.glTexture,
                    0,
                ); // level must be 0
            } else {
                gl.framebufferRenderbuffer(
                    gl.FRAMEBUFFER,
                    glAttachment,
                    gl.RENDERBUFFER,
                    dst.glRenderbuffer,
                );
            }
        }

        gl.drawBuffers(attachments);

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            switch (status) {
            case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT: {
                console.error('glCheckFramebufferStatus() - FRAMEBUFFER_INCOMPLETE_ATTACHMENT');
                break;
            }
            case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: {
                console.error('glCheckFramebufferStatus() - FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT');
                break;
            }
            case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS: {
                console.error('glCheckFramebufferStatus() - FRAMEBUFFER_INCOMPLETE_DIMENSIONS');
                break;
            }
            case gl.FRAMEBUFFER_UNSUPPORTED: {
                console.error('glCheckFramebufferStatus() - FRAMEBUFFER_UNSUPPORTED');
                break;
            }
            default:
            }
        }

        if (device.stateCache.glFramebuffer !== gpuFramebuffer.glFramebuffer) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, device.stateCache.glFramebuffer);
        }
    }
}

export function WebGPUCmdFuncDestroyFramebuffer (device: WebGPUDevice, gpuFramebuffer: IWebGPUGPUFramebuffer) {
    if (gpuFramebuffer.glFramebuffer) {
        device.gl.deleteFramebuffer(gpuFramebuffer.glFramebuffer);
        gpuFramebuffer.glFramebuffer = null;
    }
}

type ShaderStage = 'vertex' | 'fragment' | 'compute';

export function WebGPUCmdFuncCreateShader (device: WebGPUDevice, gpuShader: IWebGPUGPUShader) {
    const nativeDevice = device.nativeDevice()!;
    const glslang = device.glslang()!;

    for (let k = 0; k < gpuShader.gpuStages.length; k++) {
        const gpuStage = gpuShader.gpuStages[k];

        let glShaderType: number = GPUShaderStage.VERTEX;
        let shaderTypeStr: ShaderStage = 'vertex';
        const lineNumber = 1;

        switch (gpuStage.type) {
        case ShaderStageFlagBit.VERTEX: {
            shaderTypeStr = 'vertex';
            glShaderType = GPUShaderStage.VERTEX;
            break;
        }
        case ShaderStageFlagBit.FRAGMENT: {
            shaderTypeStr = 'fragment';
            glShaderType = GPUShaderStage.FRAGMENT;
            break;
        }
        case ShaderStageFlagBit.COMPUTE: {
            shaderTypeStr = 'compute';
            glShaderType = GPUShaderStage.COMPUTE;
            break;
        }
        default: {
            console.error('Unsupported GFXShaderType.');
            return;
        }
        }

        const useWGSL = false;
        const sourceCode = `#version 310 es\n${gpuStage.source}`;
        const code = useWGSL ? sourceCode : glslang.compileGLSL(sourceCode, shaderTypeStr, true);
        const shader: GPUShaderModule = nativeDevice?.createShaderModule({ code });
        const shaderStage: GPUProgrammableStageDescriptor = {
            module: shader,
            entryPoint: 'main',
        };
        gpuStage.glShader = shaderStage;
        // const complieInfo = shader.compilationInfo();
        // void complieInfo.then((info) => {
        //     console.info(info);
        // });

        // const glShader = gl.createShader(glShaderType);
        // if (glShader) {
        //     gpuStage.glShader = glShader;
        //     gl.shaderSource(gpuStage.glShader, '#version 300 es\n' + gpuStage.source);
        //     gl.compileShader(gpuStage.glShader);

        //     if (!gl.getShaderParameter(gpuStage.glShader, gl.COMPILE_STATUS)) {
        //         console.error(shaderTypeStr + ' in \'' + gpuShader.name + '\' compilation failed.');
        //         console.error('Shader source dump:', gpuStage.source.replace(/^|\n/g, () => `\n${lineNumber++} `));
        //         console.error(gl.getShaderInfoLog(gpuStage.glShader));

        //         for (let l = 0; l < gpuShader.gpuStages.length; l++) {
        //             const stage = gpuShader.gpuStages[k];
        //             if (stage.glShader) {
        //                 gl.deleteShader(stage.glShader);
        //                 stage.glShader = null;
        //             }
        //         }
        //         return;
        //     }
        // }
    }

    // const glProgram = gl.createProgram();
    // if (!glProgram) {
    //     return;
    // }

    // gpuShader.glProgram = glProgram;

    // // link program
    // for (let k = 0; k < gpuShader.gpuStages.length; k++) {
    //     const gpuStage = gpuShader.gpuStages[k];
    //     gl.attachShader(gpuShader.glProgram, gpuStage.glShader!);
    // }

    // gl.linkProgram(gpuShader.glProgram);

    // // detach & delete immediately
    // for (let k = 0; k < gpuShader.gpuStages.length; k++) {
    //     const gpuStage = gpuShader.gpuStages[k];
    //     if (gpuStage.glShader) {
    //         gl.detachShader(gpuShader.glProgram, gpuStage.glShader);
    //         gl.deleteShader(gpuStage.glShader);
    //         gpuStage.glShader = null;
    //     }
    // }

    // if (gl.getProgramParameter(gpuShader.glProgram, gl.LINK_STATUS)) {
    //     console.info('Shader \'' + gpuShader.name + '\' compilation succeeded.');
    // } else {
    //     console.error('Failed to link shader \'' + gpuShader.name + '\'.');
    //     console.error(gl.getProgramInfoLog(gpuShader.glProgram));
    //     return;
    // }

    // parse inputs
    const activeAttribCount = gl.getProgramParameter(gpuShader.glProgram, gl.ACTIVE_ATTRIBUTES);
    gpuShader.glInputs = new Array<IWebGPUGPUInput>(activeAttribCount);

    for (let i = 0; i < activeAttribCount; ++i) {
        const attribInfo = gl.getActiveAttrib(gpuShader.glProgram, i);
        if (attribInfo) {
            let varName: string;
            const nameOffset = attribInfo.name.indexOf('[');
            if (nameOffset !== -1) {
                varName = attribInfo.name.substr(0, nameOffset);
            } else {
                varName = attribInfo.name;
            }

            const glLoc = gl.getAttribLocation(gpuShader.glProgram, varName);
            const type = WebGLTypeToGFXType(attribInfo.type, gl);
            const stride = WebGLGetTypeSize(attribInfo.type, gl);

            gpuShader.glInputs[i] = {
                name: varName,
                type,
                stride,
                count: attribInfo.size,
                size: stride * attribInfo.size,

                glType: attribInfo.type,
                glLoc,
            };
        }
    }

    // create uniform blocks
    const activeBlockCount = gl.getProgramParameter(gpuShader.glProgram, gl.ACTIVE_UNIFORM_BLOCKS);
    let blockName: string;
    let blockIdx: number;
    let blockSize: number;
    let block: UniformBlock | null;

    if (activeBlockCount) {
        gpuShader.glBlocks = new Array<IWebGPUGPUUniformBlock>(activeBlockCount);

        for (let b = 0; b < activeBlockCount; ++b) {
            blockName = gl.getActiveUniformBlockName(gpuShader.glProgram, b)!;
            const nameOffset = blockName.indexOf('[');
            if (nameOffset !== -1) {
                blockName = blockName.substr(0, nameOffset);
            }

            // blockIdx = gl.getUniformBlockIndex(gpuShader.glProgram, blockName);
            block = null;
            for (let k = 0; k < gpuShader.blocks.length; k++) {
                if (gpuShader.blocks[k].name === blockName) {
                    block = gpuShader.blocks[k];
                    break;
                }
            }

            if (!block) {
                error(`Block '${blockName}' does not bound`);
            } else {
                // blockIdx = gl.getUniformBlockIndex(gpuShader.glProgram, blockName);
                blockIdx = b;
                blockSize = gl.getActiveUniformBlockParameter(gpuShader.glProgram, blockIdx, gl.UNIFORM_BLOCK_DATA_SIZE);
                const glBinding = block.binding + (device.bindingMappingInfo.bufferOffsets[block.set] || 0);

                gl.uniformBlockBinding(gpuShader.glProgram, blockIdx, glBinding);

                gpuShader.glBlocks[b] = {
                    set: block.set,
                    binding: block.binding,
                    idx: blockIdx,
                    name: blockName,
                    size: blockSize,
                    glBinding,
                };
            }
        }
    }

    // create uniform samplers
    if (gpuShader.samplers.length > 0) {
        gpuShader.glSamplers = new Array<IWebGPUGPUUniformSampler>(gpuShader.samplers.length);

        for (let i = 0; i < gpuShader.samplers.length; ++i) {
            const sampler = gpuShader.samplers[i];
            gpuShader.glSamplers[i] = {
                set: sampler.set,
                binding: sampler.binding,
                name: sampler.name,
                type: sampler.type,
                count: sampler.count,
                units: [],
                glUnits: null!,
                glType: GFXTypeToWebGLType(sampler.type, gl),
                glLoc: null!,
            };
        }
    }

    // texture unit index mapping optimization
    const glActiveSamplers: IWebGPUGPUUniformSampler[] = [];
    const glActiveSamplerLocations: WebGLUniformLocation[] = [];
    const bindingMappingInfo = device.bindingMappingInfo;
    const texUnitCacheMap = device.stateCache.texUnitCacheMap;

    let flexibleSetBaseOffset = 0;
    for (let i = 0; i < gpuShader.blocks.length; ++i) {
        if (gpuShader.blocks[i].set === bindingMappingInfo.flexibleSet) {
            flexibleSetBaseOffset++;
        }
    }

    let arrayOffset = 0;
    for (let i = 0; i < gpuShader.samplers.length; ++i) {
        const sampler = gpuShader.samplers[i];
        const glLoc = gl.getUniformLocation(gpuShader.glProgram, sampler.name);
        if (glLoc) {
            glActiveSamplers.push(gpuShader.glSamplers[i]);
            glActiveSamplerLocations.push(glLoc);
        }
        if (texUnitCacheMap[sampler.name] === undefined) {
            let binding = sampler.binding + bindingMappingInfo.samplerOffsets[sampler.set] + arrayOffset;
            if (sampler.set === bindingMappingInfo.flexibleSet) binding -= flexibleSetBaseOffset;
            texUnitCacheMap[sampler.name] = binding % device.maxTextureUnits;
            arrayOffset += sampler.count - 1;
        }
    }

    if (glActiveSamplers.length) {
        const usedTexUnits: boolean[] = [];
        // try to reuse existing mappings first
        for (let i = 0; i < glActiveSamplers.length; ++i) {
            const glSampler = glActiveSamplers[i];

            let cachedUnit = texUnitCacheMap[glSampler.name];
            if (cachedUnit !== undefined) {
                glSampler.glLoc = glActiveSamplerLocations[i];
                for (let t = 0; t < glSampler.count; ++t) {
                    while (usedTexUnits[cachedUnit]) {
                        cachedUnit = (cachedUnit + 1) % device.maxTextureUnits;
                    }
                    glSampler.units.push(cachedUnit);
                    usedTexUnits[cachedUnit] = true;
                }
            }
        }
        // fill in the rest sequencially
        let unitIdx = 0;
        for (let i = 0; i < glActiveSamplers.length; ++i) {
            const glSampler = glActiveSamplers[i];

            if (!glSampler.glLoc) {
                glSampler.glLoc = glActiveSamplerLocations[i];
                while (usedTexUnits[unitIdx]) unitIdx++;
                for (let t = 0; t < glSampler.count; ++t) {
                    while (usedTexUnits[unitIdx]) {
                        unitIdx = (unitIdx + 1) % device.maxTextureUnits;
                    }
                    if (texUnitCacheMap[glSampler.name] === undefined) {
                        texUnitCacheMap[glSampler.name] = unitIdx;
                    }
                    glSampler.units.push(unitIdx);
                    usedTexUnits[unitIdx] = true;
                }
            }
        }

        if (device.stateCache.glProgram !== gpuShader.glProgram) {
            gl.useProgram(gpuShader.glProgram);
        }

        for (let k = 0; k < glActiveSamplers.length; k++) {
            const glSampler = glActiveSamplers[k];
            glSampler.glUnits = new Int32Array(glSampler.units);
            gl.uniform1iv(glSampler.glLoc, glSampler.glUnits);
        }

        if (device.stateCache.glProgram !== gpuShader.glProgram) {
            gl.useProgram(device.stateCache.glProgram);
        }
    }

    gpuShader.glSamplers = glActiveSamplers;
}

export function WebGPUCmdFuncDestroyShader (device: WebGPUDevice, gpuShader: IWebGPUGPUShader) {
    if (gpuShader.glProgram) {
        device.gl.deleteProgram(gpuShader.glProgram);
        gpuShader.glProgram = null;
    }
}

export function WebGPUCmdFuncCreateInputAssember (device: WebGPUDevice, gpuInputAssembler: IWebGPUGPUInputAssembler) {
    gpuInputAssembler.glAttribs = new Array<IWebGPUAttrib>(gpuInputAssembler.attributes.length);

    const offsets = [0, 0, 0, 0, 0, 0, 0, 0];

    for (let i = 0; i < gpuInputAssembler.attributes.length; ++i) {
        const attrib = gpuInputAssembler.attributes[i];

        const stream = attrib.stream !== undefined ? attrib.stream : 0;
        // if (stream < gpuInputAssembler.gpuVertexBuffers.length) {

        const gpuBuffer = gpuInputAssembler.gpuVertexBuffers[stream];

        const glType = 0;
        const size = FormatInfos[attrib.format].size;

        gpuInputAssembler.glAttribs[i] = {
            name: attrib.name,
            glBuffer: gpuBuffer.glBuffer,
            glType,
            size,
            count: FormatInfos[attrib.format].count,
            stride: gpuBuffer.stride,
            componentCount: 4,
            isNormalized: (attrib.isNormalized !== undefined ? attrib.isNormalized : false),
            isInstanced: (attrib.isInstanced !== undefined ? attrib.isInstanced : false),
            offset: offsets[stream],
        };

        offsets[stream] += size;
    }
}

export function WebGPUCmdFuncDestroyInputAssembler (device: WebGPUDevice, gpuInputAssembler: IWebGPUGPUInputAssembler) {
    const it = gpuInputAssembler.glVAOs.values();
    let res = it.next();
    while (!res.done) {
        device.gl.deleteVertexArray(res.value);
        res = it.next();
    }
    gpuInputAssembler.glVAOs.clear();
}

interface IWebGPUStateCache {
    gpuPipelineState: IWebGPUGPUPipelineState | null;
    gpuInputAssembler: IWebGPUGPUInputAssembler | null;
    reverseCW: boolean;
    glPrimitive: GPUPrimitiveTopology;
    invalidateAttachments: GLenum[];
}
const gfxStateCache: IWebGPUStateCache = {
    gpuPipelineState: null,
    gpuInputAssembler: null,
    reverseCW: false,
    glPrimitive: 'triangle-list',
    invalidateAttachments: [],
};

export function WebGPUCmdFuncBeginRenderPass (
    device: WebGPUDevice,
    gpuRenderPass: WebGPURenderPass,
    gpuFramebuffer: IWebGPUGPUFramebuffer | null,
    renderArea: Rect,
    clearColors: Color[],
    clearDepth: number,
    clearStencil: number,
) {
    const nativeDevice = device.nativeDevice();

    const cache = device.stateCache;

    let clears: GLbitfield = 0;

    const nativeRenPassDesc = gpuRenderPass.gpuRenderPass.nativeRenderPass!;
    for (let i = 0; i < clearColors.length; i++) {
        nativeRenPassDesc.colorAttachments[i] = {
            attachment: gpuFramebuffer?.gpuColorTextures[i].glTexture,
            loadValue: clearColors[i], // ABGR/RGBA/BGRA ?
            storeOp: gpuRenderPass?.colorAttachments[i].storeOp === StoreOp.STORE ? 'store' : 'clear',
        };
    }

    if (gpuRenderPass?.depthStencilAttachment) {
        nativeRenPassDesc.depthStencilAttachment = {
            attachment: gpuFramebuffer?.gpuDepthStencilTexture?.glTexture?.createView() as GPUTextureView,
            depthLoadValue: clearDepth,
            depthStoreOp: gpuRenderPass.depthStencilAttachment.depthStoreOp === StoreOp.STORE ? 'store' : 'clear',
            // depthReadOnly:
            stencilLoadValue: clearStencil,
            stencilStoreOp: gpuRenderPass.depthStencilAttachment.stencilStoreOp === StoreOp.STORE ? 'store' : 'clear',
            // stencilReadOnly:
        };
    }

    if (gpuFramebuffer && gpuRenderPass) {
        if (cache.glFramebuffer !== gpuFramebuffer.glFramebuffer) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, gpuFramebuffer.glFramebuffer);
            cache.glFramebuffer = gpuFramebuffer.glFramebuffer;
            // render targets are drawn with flipped-Y
            const reverseCW = !!gpuFramebuffer.glFramebuffer;
            if (reverseCW !== gfxStateCache.reverseCW) {
                gfxStateCache.reverseCW = reverseCW;
                const isCCW = !device.stateCache.rs.isFrontFaceCCW;
                gl.frontFace(isCCW ? gl.CCW : gl.CW);
                device.stateCache.rs.isFrontFaceCCW = isCCW;
            }
        }

        if (cache.viewport.left !== renderArea.x
            || cache.viewport.top !== renderArea.y
            || cache.viewport.width !== renderArea.width
            || cache.viewport.height !== renderArea.height) {
            gl.viewport(renderArea.x, renderArea.y, renderArea.width, renderArea.height);

            cache.viewport.left = renderArea.x;
            cache.viewport.top = renderArea.y;
            cache.viewport.width = renderArea.width;
            cache.viewport.height = renderArea.height;
        }

        if (cache.scissorRect.x !== renderArea.x
            || cache.scissorRect.y !== renderArea.y
            || cache.scissorRect.width !== renderArea.width
            || cache.scissorRect.height !== renderArea.height) {
            gl.scissor(renderArea.x, renderArea.y, renderArea.width, renderArea.height);

            cache.scissorRect.x = renderArea.x;
            cache.scissorRect.y = renderArea.y;
            cache.scissorRect.width = renderArea.width;
            cache.scissorRect.height = renderArea.height;
        }

        gfxStateCache.invalidateAttachments.length = 0;

        for (let j = 0; j < clearColors.length; ++j) {
            const colorAttachment = gpuRenderPass.colorAttachments[j];

            if (colorAttachment.format !== Format.UNKNOWN) {
                switch (colorAttachment.loadOp) {
                case LoadOp.LOAD: break; // GL default behavior
                case LoadOp.CLEAR: {
                    if (cache.bs.targets[0].blendColorMask !== ColorMask.ALL) {
                        gl.colorMask(true, true, true, true);
                    }

                    if (!gpuFramebuffer.isOffscreen) {
                        const clearColor = clearColors[0];
                        gl.clearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
                        clears |= gl.COLOR_BUFFER_BIT;
                    } else {
                        _f32v4[0] = clearColors[j].x;
                        _f32v4[1] = clearColors[j].y;
                        _f32v4[2] = clearColors[j].z;
                        _f32v4[3] = clearColors[j].w;
                        gl.clearBufferfv(gl.COLOR, j, _f32v4);
                    }
                    break;
                }
                case LoadOp.DISCARD: {
                    // invalidate the framebuffer
                    gfxStateCache.invalidateAttachments.push(gl.COLOR_ATTACHMENT0 + j);
                    break;
                }
                default:
                }
            }
        } // if (curGPURenderPass)

        if (gpuRenderPass.depthStencilAttachment) {
            if (gpuRenderPass.depthStencilAttachment.format !== Format.UNKNOWN) {
                switch (gpuRenderPass.depthStencilAttachment.depthLoadOp) {
                case LoadOp.LOAD: break; // GL default behavior
                case LoadOp.CLEAR: {
                    if (!cache.dss.depthWrite) {
                        gl.depthMask(true);
                    }

                    gl.clearDepth(clearDepth);

                    clears |= gl.DEPTH_BUFFER_BIT;
                    break;
                }
                case LoadOp.DISCARD: {
                    // invalidate the framebuffer
                    gfxStateCache.invalidateAttachments.push(gl.DEPTH_ATTACHMENT);
                    break;
                }
                default:
                }

                if (FormatInfos[gpuRenderPass.depthStencilAttachment.format].hasStencil) {
                    switch (gpuRenderPass.depthStencilAttachment.stencilLoadOp) {
                    case LoadOp.LOAD: break; // GL default behavior
                    case LoadOp.CLEAR: {
                        if (!cache.dss.stencilWriteMaskFront) {
                            gl.stencilMaskSeparate(gl.FRONT, 0xffff);
                        }

                        if (!cache.dss.stencilWriteMaskBack) {
                            gl.stencilMaskSeparate(gl.BACK, 0xffff);
                        }

                        gl.clearStencil(clearStencil);
                        clears |= gl.STENCIL_BUFFER_BIT;
                        break;
                    }
                    case LoadOp.DISCARD: {
                        // invalidate the framebuffer
                        gfxStateCache.invalidateAttachments.push(gl.STENCIL_ATTACHMENT);
                        break;
                    }
                    default:
                    }
                }
            }
        } // if (curGPURenderPass.depthStencilAttachment)

        if (gpuFramebuffer.glFramebuffer && gfxStateCache.invalidateAttachments.length) {
            gl.invalidateFramebuffer(gl.FRAMEBUFFER, gfxStateCache.invalidateAttachments);
        }

        if (clears) {
            gl.clear(clears);
        }

        // restore states
        if (clears & gl.COLOR_BUFFER_BIT) {
            const colorMask = cache.bs.targets[0].blendColorMask;
            if (colorMask !== ColorMask.ALL) {
                const r = (colorMask & ColorMask.R) !== ColorMask.NONE;
                const g = (colorMask & ColorMask.G) !== ColorMask.NONE;
                const b = (colorMask & ColorMask.B) !== ColorMask.NONE;
                const a = (colorMask & ColorMask.A) !== ColorMask.NONE;
                gl.colorMask(r, g, b, a);
            }
        }

        if ((clears & gl.DEPTH_BUFFER_BIT)
            && !cache.dss.depthWrite) {
            gl.depthMask(false);
        }

        if (clears & gl.STENCIL_BUFFER_BIT) {
            if (!cache.dss.stencilWriteMaskFront) {
                gl.stencilMaskSeparate(gl.FRONT, 0);
            }

            if (!cache.dss.stencilWriteMaskBack) {
                gl.stencilMaskSeparate(gl.BACK, 0);
            }
        }
    } // if (gpuFramebuffer)
}

export function WebGPUCmdFuncBindStates (
    device: WebGPUDevice,
    gpuPipelineState: IWebGPUGPUPipelineState | null,
    gpuInputAssembler: IWebGPUGPUInputAssembler | null,
    gpuDescriptorSets: IWebGPUGPUDescriptorSet[],
    dynamicOffsets: number[],
    viewport: Viewport | null,
    scissor: Rect | null,
    lineWidth: number | null,
    depthBias: IWebGPUDepthBias | null,
    blendConstants: number[],
    depthBounds: IWebGPUDepthBounds | null,
    stencilWriteMask: IWebGPUStencilWriteMask | null,
    stencilCompareMask: IWebGPUStencilCompareMask | null,
) {
    const gl = device.gl;
    const cache = device.stateCache;
    const gpuShader = gpuPipelineState && gpuPipelineState.gpuShader;

    const isShaderChanged = false;

    // bind pipeline
    if (gpuPipelineState && gfxStateCache.gpuPipelineState !== gpuPipelineState) {
        gfxStateCache.gpuPipelineState = gpuPipelineState;
        gfxStateCache.glPrimitive = gpuPipelineState.glPrimitive;

        // if (gpuShader) {
        //     const glProgram = gpuShader.glProgram;
        //     if (cache.glProgram !== glProgram) {
        //         gl.useProgram(glProgram);
        //         cache.glProgram = glProgram;
        //         isShaderChanged = true;
        //     }
        // }

        // rasterizer state
        const rs = gpuPipelineState.rs;
        if (rs) {
            if (cache.rs.cullMode !== rs.cullMode) {
                switch (rs.cullMode) {
                case CullMode.NONE: {
                    gl.disable(gl.CULL_FACE);
                    break;
                }
                case CullMode.FRONT: {
                    gl.enable(gl.CULL_FACE);
                    gl.cullFace(gl.FRONT);
                    break;
                }
                case CullMode.BACK: {
                    gl.enable(gl.CULL_FACE);
                    gl.cullFace(gl.BACK);
                    break;
                }
                default:
                }

                device.stateCache.rs.cullMode = rs.cullMode;
            }

            const isFrontFaceCCW = rs.isFrontFaceCCW !== gfxStateCache.reverseCW; // boolean XOR
            if (device.stateCache.rs.isFrontFaceCCW !== isFrontFaceCCW) {
                gl.frontFace(isFrontFaceCCW ? gl.CCW : gl.CW);
                device.stateCache.rs.isFrontFaceCCW = isFrontFaceCCW;
            }

            if ((device.stateCache.rs.depthBias !== rs.depthBias)
                || (device.stateCache.rs.depthBiasSlop !== rs.depthBiasSlop)) {
                gl.polygonOffset(rs.depthBias, rs.depthBiasSlop);
                device.stateCache.rs.depthBias = rs.depthBias;
                device.stateCache.rs.depthBiasSlop = rs.depthBiasSlop;
            }

            if (device.stateCache.rs.lineWidth !== rs.lineWidth) {
                gl.lineWidth(rs.lineWidth);
                device.stateCache.rs.lineWidth = rs.lineWidth;
            }
        } // rasterizater state

        // depth-stencil state
        const dss = gpuPipelineState.dss;
        if (dss) {
            if (cache.dss.depthTest !== dss.depthTest) {
                if (dss.depthTest) {
                    gl.enable(gl.DEPTH_TEST);
                } else {
                    gl.disable(gl.DEPTH_TEST);
                }
                cache.dss.depthTest = dss.depthTest;
            }

            if (cache.dss.depthWrite !== dss.depthWrite) {
                gl.depthMask(dss.depthWrite);
                cache.dss.depthWrite = dss.depthWrite;
            }

            if (cache.dss.depthFunc !== dss.depthFunc) {
                gl.depthFunc(WebGLCmpFuncs[dss.depthFunc]);
                cache.dss.depthFunc = dss.depthFunc;
            }

            // front
            if ((cache.dss.stencilTestFront !== dss.stencilTestFront)
                || (cache.dss.stencilTestBack !== dss.stencilTestBack)) {
                if (dss.stencilTestFront || dss.stencilTestBack) {
                    gl.enable(gl.STENCIL_TEST);
                } else {
                    gl.disable(gl.STENCIL_TEST);
                }
                cache.dss.stencilTestFront = dss.stencilTestFront;
                cache.dss.stencilTestBack = dss.stencilTestBack;
            }

            if ((cache.dss.stencilFuncFront !== dss.stencilFuncFront)
                || (cache.dss.stencilRefFront !== dss.stencilRefFront)
                || (cache.dss.stencilReadMaskFront !== dss.stencilReadMaskFront)) {
                gl.stencilFuncSeparate(
                    gl.FRONT,
                    WebGLCmpFuncs[dss.stencilFuncFront],
                    dss.stencilRefFront,
                    dss.stencilReadMaskFront,
                );

                cache.dss.stencilFuncFront = dss.stencilFuncFront;
                cache.dss.stencilRefFront = dss.stencilRefFront;
                cache.dss.stencilReadMaskFront = dss.stencilReadMaskFront;
            }

            if ((cache.dss.stencilFailOpFront !== dss.stencilFailOpFront)
                || (cache.dss.stencilZFailOpFront !== dss.stencilZFailOpFront)
                || (cache.dss.stencilPassOpFront !== dss.stencilPassOpFront)) {
                gl.stencilOpSeparate(
                    gl.FRONT,
                    WebGLStencilOps[dss.stencilFailOpFront],
                    WebGLStencilOps[dss.stencilZFailOpFront],
                    WebGLStencilOps[dss.stencilPassOpFront],
                );

                cache.dss.stencilFailOpFront = dss.stencilFailOpFront;
                cache.dss.stencilZFailOpFront = dss.stencilZFailOpFront;
                cache.dss.stencilPassOpFront = dss.stencilPassOpFront;
            }

            if (cache.dss.stencilWriteMaskFront !== dss.stencilWriteMaskFront) {
                gl.stencilMaskSeparate(gl.FRONT, dss.stencilWriteMaskFront);
                cache.dss.stencilWriteMaskFront = dss.stencilWriteMaskFront;
            }

            // back
            if ((cache.dss.stencilFuncBack !== dss.stencilFuncBack)
                || (cache.dss.stencilRefBack !== dss.stencilRefBack)
                || (cache.dss.stencilReadMaskBack !== dss.stencilReadMaskBack)) {
                gl.stencilFuncSeparate(
                    gl.BACK,
                    WebGLCmpFuncs[dss.stencilFuncBack],
                    dss.stencilRefBack,
                    dss.stencilReadMaskBack,
                );

                cache.dss.stencilFuncBack = dss.stencilFuncBack;
                cache.dss.stencilRefBack = dss.stencilRefBack;
                cache.dss.stencilReadMaskBack = dss.stencilReadMaskBack;
            }

            if ((cache.dss.stencilFailOpBack !== dss.stencilFailOpBack)
                || (cache.dss.stencilZFailOpBack !== dss.stencilZFailOpBack)
                || (cache.dss.stencilPassOpBack !== dss.stencilPassOpBack)) {
                gl.stencilOpSeparate(
                    gl.BACK,
                    WebGLStencilOps[dss.stencilFailOpBack],
                    WebGLStencilOps[dss.stencilZFailOpBack],
                    WebGLStencilOps[dss.stencilPassOpBack],
                );

                cache.dss.stencilFailOpBack = dss.stencilFailOpBack;
                cache.dss.stencilZFailOpBack = dss.stencilZFailOpBack;
                cache.dss.stencilPassOpBack = dss.stencilPassOpBack;
            }

            if (cache.dss.stencilWriteMaskBack !== dss.stencilWriteMaskBack) {
                gl.stencilMaskSeparate(gl.BACK, dss.stencilWriteMaskBack);
                cache.dss.stencilWriteMaskBack = dss.stencilWriteMaskBack;
            }
        } // depth-stencil state

        // blend state
        const bs = gpuPipelineState.bs;
        if (bs) {
            if (cache.bs.isA2C !== bs.isA2C) {
                if (bs.isA2C) {
                    gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);
                } else {
                    gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
                }
                cache.bs.isA2C = bs.isA2C;
            }

            if ((cache.bs.blendColor.x !== bs.blendColor.x)
                || (cache.bs.blendColor.y !== bs.blendColor.y)
                || (cache.bs.blendColor.z !== bs.blendColor.z)
                || (cache.bs.blendColor.w !== bs.blendColor.w)) {
                gl.blendColor(bs.blendColor.x, bs.blendColor.y, bs.blendColor.z, bs.blendColor.w);

                cache.bs.blendColor.x = bs.blendColor.x;
                cache.bs.blendColor.y = bs.blendColor.y;
                cache.bs.blendColor.z = bs.blendColor.z;
                cache.bs.blendColor.w = bs.blendColor.w;
            }

            const target0 = bs.targets[0];
            const target0Cache = cache.bs.targets[0];

            if (target0Cache.blend !== target0.blend) {
                if (target0.blend) {
                    gl.enable(gl.BLEND);
                } else {
                    gl.disable(gl.BLEND);
                }
                target0Cache.blend = target0.blend;
            }

            if ((target0Cache.blendEq !== target0.blendEq)
                || (target0Cache.blendAlphaEq !== target0.blendAlphaEq)) {
                gl.blendEquationSeparate(WebGLBlendOps[target0.blendEq], WebGLBlendOps[target0.blendAlphaEq]);
                target0Cache.blendEq = target0.blendEq;
                target0Cache.blendAlphaEq = target0.blendAlphaEq;
            }

            if ((target0Cache.blendSrc !== target0.blendSrc)
                || (target0Cache.blendDst !== target0.blendDst)
                || (target0Cache.blendSrcAlpha !== target0.blendSrcAlpha)
                || (target0Cache.blendDstAlpha !== target0.blendDstAlpha)) {
                gl.blendFuncSeparate(
                    WebGLBlendFactors[target0.blendSrc],
                    WebGLBlendFactors[target0.blendDst],
                    WebGLBlendFactors[target0.blendSrcAlpha],
                    WebGLBlendFactors[target0.blendDstAlpha],
                );

                target0Cache.blendSrc = target0.blendSrc;
                target0Cache.blendDst = target0.blendDst;
                target0Cache.blendSrcAlpha = target0.blendSrcAlpha;
                target0Cache.blendDstAlpha = target0.blendDstAlpha;
            }

            if (target0Cache.blendColorMask !== target0.blendColorMask) {
                gl.colorMask(
                    (target0.blendColorMask & ColorMask.R) !== ColorMask.NONE,
                    (target0.blendColorMask & ColorMask.G) !== ColorMask.NONE,
                    (target0.blendColorMask & ColorMask.B) !== ColorMask.NONE,
                    (target0.blendColorMask & ColorMask.A) !== ColorMask.NONE,
                );

                target0Cache.blendColorMask = target0.blendColorMask;
            }
        } // blend state
    } // bind pipeline

    // bind descriptor sets
    if (gpuPipelineState && gpuPipelineState.gpuPipelineLayout && gpuShader) {
        const blockLen = gpuShader.glBlocks.length;
        const dynamicOffsetIndices = gpuPipelineState.gpuPipelineLayout.dynamicOffsetIndices;

        for (let j = 0; j < blockLen; j++) {
            const glBlock = gpuShader.glBlocks[j];
            const gpuDescriptorSet = gpuDescriptorSets[glBlock.set];
            const descriptorIndex = gpuDescriptorSet && gpuDescriptorSet.descriptorIndices[glBlock.binding];
            const gpuDescriptor = descriptorIndex >= 0 && gpuDescriptorSet.gpuDescriptors[descriptorIndex];

            if (!gpuDescriptor || !gpuDescriptor.gpuBuffer) {
                error(`Buffer binding '${glBlock.name}' at set ${glBlock.set} binding ${glBlock.binding} is not bounded`);
                continue;
            }

            const dynamicOffsetIndexSet = dynamicOffsetIndices[glBlock.set];
            const dynamicOffsetIndex = dynamicOffsetIndexSet && dynamicOffsetIndexSet[glBlock.binding];
            let offset = gpuDescriptor.gpuBuffer.glOffset;
            if (dynamicOffsetIndex >= 0) offset += dynamicOffsets[dynamicOffsetIndex];

            if (cache.glBindUBOs[glBlock.glBinding] !== gpuDescriptor.gpuBuffer.glBuffer
                || cache.glBindUBOOffsets[glBlock.glBinding] !== offset) {
                if (offset) {
                    gl.bindBufferRange(gl.UNIFORM_BUFFER, glBlock.glBinding, gpuDescriptor.gpuBuffer.glBuffer,
                        offset, gpuDescriptor.gpuBuffer.size);
                } else {
                    gl.bindBufferBase(gl.UNIFORM_BUFFER, glBlock.glBinding, gpuDescriptor.gpuBuffer.glBuffer);
                }
                cache.glUniformBuffer = cache.glBindUBOs[glBlock.glBinding] = gpuDescriptor.gpuBuffer.glBuffer;
                cache.glBindUBOOffsets[glBlock.glBinding] = offset;
            }
        }

        const samplerLen = gpuShader.glSamplers.length;
        for (let i = 0; i < samplerLen; i++) {
            const glSampler = gpuShader.glSamplers[i];
            const gpuDescriptorSet = gpuDescriptorSets[glSampler.set];
            let descriptorIndex = gpuDescriptorSet && gpuDescriptorSet.descriptorIndices[glSampler.binding];
            let gpuDescriptor = descriptorIndex >= 0 && gpuDescriptorSet.gpuDescriptors[descriptorIndex];

            for (let l = 0; l < glSampler.units.length; l++) {
                const texUnit = glSampler.units[l];

                const glTexUnit = cache.glTexUnits[texUnit];

                if (!gpuDescriptor || !gpuDescriptor.gpuTexture || !gpuDescriptor.gpuSampler) {
                    error(`Sampler binding '${glSampler.name}' at set ${glSampler.set} binding ${glSampler.binding} index ${l} is not bounded`);
                    continue;
                }

                if (gpuDescriptor.gpuTexture
                    && gpuDescriptor.gpuTexture.size > 0) {
                    const gpuTexture = gpuDescriptor.gpuTexture;
                    if (glTexUnit.glTexture !== gpuTexture.glTexture) {
                        if (cache.texUnit !== texUnit) {
                            gl.activeTexture(gl.TEXTURE0 + texUnit);
                            cache.texUnit = texUnit;
                        }
                        if (gpuTexture.glTexture) {
                            gl.bindTexture(gpuTexture.glTarget, gpuTexture.glTexture);
                        } else {
                            gl.bindTexture(gpuTexture.glTarget, device.nullTex2D!.gpuTexture.glTexture);
                        }
                        glTexUnit.glTexture = gpuTexture.glTexture;
                    }

                    const gpuSampler = gpuDescriptor.gpuSampler;
                    if (cache.glSamplerUnits[texUnit] !== gpuSampler.glSampler) {
                        gl.bindSampler(texUnit, gpuSampler.glSampler);
                        cache.glSamplerUnits[texUnit] = gpuSampler.glSampler;
                    }
                }

                gpuDescriptor = gpuDescriptorSet.gpuDescriptors[++descriptorIndex];
            }
        }
    } // bind descriptor sets

    // bind vertex/index buffer
    if (gpuInputAssembler && gpuShader
        && (isShaderChanged || gfxStateCache.gpuInputAssembler !== gpuInputAssembler)) {
        gfxStateCache.gpuInputAssembler = gpuInputAssembler;

        if (device.useVAO) {
            // check vao
            let glVAO = gpuInputAssembler.glVAOs.get(gpuShader.glProgram!);
            if (!glVAO) {
                glVAO = gl.createVertexArray()!;
                gpuInputAssembler.glVAOs.set(gpuShader.glProgram!, glVAO);

                gl.bindVertexArray(glVAO);
                gl.bindBuffer(gl.ARRAY_BUFFER, null);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
                cache.glArrayBuffer = null;
                cache.glElementArrayBuffer = null;

                let glAttrib: IWebGPUAttrib | null;
                for (let j = 0; j < gpuShader.glInputs.length; j++) {
                    const glInput = gpuShader.glInputs[j];
                    glAttrib = null;

                    for (let k = 0; k < gpuInputAssembler.glAttribs.length; k++) {
                        const attrib = gpuInputAssembler.glAttribs[k];
                        if (attrib.name === glInput.name) {
                            glAttrib = attrib;
                            break;
                        }
                    }

                    if (glAttrib) {
                        if (cache.glArrayBuffer !== glAttrib.glBuffer) {
                            gl.bindBuffer(gl.ARRAY_BUFFER, glAttrib.glBuffer);
                            cache.glArrayBuffer = glAttrib.glBuffer;
                        }

                        for (let c = 0; c < glAttrib.componentCount; ++c) {
                            const glLoc = glInput.glLoc + c;
                            const attribOffset = glAttrib.offset + glAttrib.size * c;

                            gl.enableVertexAttribArray(glLoc);
                            cache.glCurrentAttribLocs[glLoc] = true;

                            gl.vertexAttribPointer(glLoc, glAttrib.count, glAttrib.glType, glAttrib.isNormalized, glAttrib.stride, attribOffset);
                            gl.vertexAttribDivisor(glLoc, glAttrib.isInstanced ? 1 : 0);
                        }
                    }
                }

                const gpuBuffer = gpuInputAssembler.gpuIndexBuffer;
                if (gpuBuffer) {
                    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.glBuffer);
                }

                gl.bindVertexArray(null);
                gl.bindBuffer(gl.ARRAY_BUFFER, null);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
                cache.glArrayBuffer = null;
                cache.glElementArrayBuffer = null;
            }

            if (cache.glVAO !== glVAO) {
                gl.bindVertexArray(glVAO);
                cache.glVAO = glVAO;
            }
        } else {
            for (let a = 0; a < device.maxVertexAttributes; ++a) {
                cache.glCurrentAttribLocs[a] = false;
            }

            for (let j = 0; j < gpuShader.glInputs.length; j++) {
                const glInput = gpuShader.glInputs[j];
                let glAttrib: IWebGPUAttrib | null = null;

                for (let k = 0; k < gpuInputAssembler.glAttribs.length; k++) {
                    const attrib = gpuInputAssembler.glAttribs[k];
                    if (attrib.name === glInput.name) {
                        glAttrib = attrib;
                        break;
                    }
                }

                if (glAttrib) {
                    if (cache.glArrayBuffer !== glAttrib.glBuffer) {
                        gl.bindBuffer(gl.ARRAY_BUFFER, glAttrib.glBuffer);
                        cache.glArrayBuffer = glAttrib.glBuffer;
                    }

                    for (let c = 0; c < glAttrib.componentCount; ++c) {
                        const glLoc = glInput.glLoc + c;
                        const attribOffset = glAttrib.offset + glAttrib.size * c;

                        if (!cache.glEnabledAttribLocs[glLoc] && glLoc >= 0) {
                            gl.enableVertexAttribArray(glLoc);
                            cache.glEnabledAttribLocs[glLoc] = true;
                        }
                        cache.glCurrentAttribLocs[glLoc] = true;

                        gl.vertexAttribPointer(glLoc, glAttrib.count, glAttrib.glType, glAttrib.isNormalized, glAttrib.stride, attribOffset);
                        gl.vertexAttribDivisor(glLoc, glAttrib.isInstanced ? 1 : 0);
                    }
                }
            } // for

            const gpuBuffer = gpuInputAssembler.gpuIndexBuffer;
            if (gpuBuffer) {
                if (cache.glElementArrayBuffer !== gpuBuffer.glBuffer) {
                    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuBuffer.glBuffer);
                    cache.glElementArrayBuffer = gpuBuffer.glBuffer;
                }
            }

            for (let a = 0; a < device.maxVertexAttributes; ++a) {
                if (cache.glEnabledAttribLocs[a] !== cache.glCurrentAttribLocs[a]) {
                    gl.disableVertexAttribArray(a);
                    cache.glEnabledAttribLocs[a] = false;
                }
            }
        }
    } // bind vertex/index buffer

    // update dynamic states
    if (gpuPipelineState && gpuPipelineState.dynamicStates.length) {
        const dsLen = gpuPipelineState.dynamicStates.length;
        for (let k = 0; k < dsLen; k++) {
            const dynamicState = gpuPipelineState.dynamicStates[k];
            switch (dynamicState) {
            case DynamicStateFlagBit.VIEWPORT: {
                if (viewport) {
                    if (cache.viewport.left !== viewport.left
                            || cache.viewport.top !== viewport.top
                            || cache.viewport.width !== viewport.width
                            || cache.viewport.height !== viewport.height) {
                        gl.viewport(viewport.left, viewport.top, viewport.width, viewport.height);

                        cache.viewport.left = viewport.left;
                        cache.viewport.top = viewport.top;
                        cache.viewport.width = viewport.width;
                        cache.viewport.height = viewport.height;
                    }
                }
                break;
            }
            case DynamicStateFlagBit.SCISSOR: {
                if (scissor) {
                    if (cache.scissorRect.x !== scissor.x
                            || cache.scissorRect.y !== scissor.y
                            || cache.scissorRect.width !== scissor.width
                            || cache.scissorRect.height !== scissor.height) {
                        gl.scissor(scissor.x, scissor.y, scissor.width, scissor.height);

                        cache.scissorRect.x = scissor.x;
                        cache.scissorRect.y = scissor.y;
                        cache.scissorRect.width = scissor.width;
                        cache.scissorRect.height = scissor.height;
                    }
                }
                break;
            }
            case DynamicStateFlagBit.LINE_WIDTH: {
                if (lineWidth) {
                    if (cache.rs.lineWidth !== lineWidth) {
                        gl.lineWidth(lineWidth);
                        cache.rs.lineWidth = lineWidth;
                    }
                }
                break;
            }
            case DynamicStateFlagBit.DEPTH_BIAS: {
                if (depthBias) {
                    if ((cache.rs.depthBias !== depthBias.constantFactor)
                            || (cache.rs.depthBiasSlop !== depthBias.slopeFactor)) {
                        gl.polygonOffset(depthBias.constantFactor, depthBias.slopeFactor);
                        cache.rs.depthBias = depthBias.constantFactor;
                        cache.rs.depthBiasSlop = depthBias.slopeFactor;
                    }
                }
                break;
            }
            case DynamicStateFlagBit.BLEND_CONSTANTS: {
                if ((cache.bs.blendColor.x !== blendConstants[0])
                        || (cache.bs.blendColor.y !== blendConstants[1])
                        || (cache.bs.blendColor.z !== blendConstants[2])
                        || (cache.bs.blendColor.w !== blendConstants[3])) {
                    gl.blendColor(blendConstants[0], blendConstants[1], blendConstants[2], blendConstants[3]);

                    cache.bs.blendColor.x = blendConstants[0];
                    cache.bs.blendColor.y = blendConstants[1];
                    cache.bs.blendColor.z = blendConstants[2];
                    cache.bs.blendColor.w = blendConstants[3];
                }
                break;
            }
            case DynamicStateFlagBit.STENCIL_WRITE_MASK: {
                if (stencilWriteMask) {
                    switch (stencilWriteMask.face) {
                    case StencilFace.FRONT: {
                        if (cache.dss.stencilWriteMaskFront !== stencilWriteMask.writeMask) {
                            gl.stencilMaskSeparate(gl.FRONT, stencilWriteMask.writeMask);
                            cache.dss.stencilWriteMaskFront = stencilWriteMask.writeMask;
                        }
                        break;
                    }
                    case StencilFace.BACK: {
                        if (cache.dss.stencilWriteMaskBack !== stencilWriteMask.writeMask) {
                            gl.stencilMaskSeparate(gl.BACK, stencilWriteMask.writeMask);
                            cache.dss.stencilWriteMaskBack = stencilWriteMask.writeMask;
                        }
                        break;
                    }
                    case StencilFace.ALL: {
                        if (cache.dss.stencilWriteMaskFront !== stencilWriteMask.writeMask
                                    || cache.dss.stencilWriteMaskBack !== stencilWriteMask.writeMask) {
                            gl.stencilMask(stencilWriteMask.writeMask);
                            cache.dss.stencilWriteMaskFront = stencilWriteMask.writeMask;
                            cache.dss.stencilWriteMaskBack = stencilWriteMask.writeMask;
                        }
                        break;
                    }
                    }
                }
                break;
            }
            case DynamicStateFlagBit.STENCIL_COMPARE_MASK: {
                if (stencilCompareMask) {
                    switch (stencilCompareMask.face) {
                    case StencilFace.FRONT: {
                        if (cache.dss.stencilRefFront !== stencilCompareMask.reference
                                    || cache.dss.stencilReadMaskFront !== stencilCompareMask.compareMask) {
                            gl.stencilFuncSeparate(
                                gl.FRONT,
                                WebGLCmpFuncs[cache.dss.stencilFuncFront],
                                stencilCompareMask.reference,
                                stencilCompareMask.compareMask,
                            );
                            cache.dss.stencilRefFront = stencilCompareMask.reference;
                            cache.dss.stencilReadMaskFront = stencilCompareMask.compareMask;
                        }
                        break;
                    }
                    case StencilFace.BACK: {
                        if (cache.dss.stencilRefBack !== stencilCompareMask.reference
                                    || cache.dss.stencilReadMaskBack !== stencilCompareMask.compareMask) {
                            gl.stencilFuncSeparate(
                                gl.BACK,
                                WebGLCmpFuncs[cache.dss.stencilFuncBack],
                                stencilCompareMask.reference,
                                stencilCompareMask.compareMask,
                            );
                            cache.dss.stencilRefBack = stencilCompareMask.reference;
                            cache.dss.stencilReadMaskBack = stencilCompareMask.compareMask;
                        }
                        break;
                    }
                    case StencilFace.ALL: {
                        if (cache.dss.stencilRefFront !== stencilCompareMask.reference
                                    || cache.dss.stencilReadMaskFront !== stencilCompareMask.compareMask
                                    || cache.dss.stencilRefBack !== stencilCompareMask.reference
                                    || cache.dss.stencilReadMaskBack !== stencilCompareMask.compareMask) {
                            gl.stencilFunc(
                                WebGLCmpFuncs[cache.dss.stencilFuncBack],
                                stencilCompareMask.reference,
                                stencilCompareMask.compareMask,
                            );
                            cache.dss.stencilRefFront = stencilCompareMask.reference;
                            cache.dss.stencilReadMaskFront = stencilCompareMask.compareMask;
                            cache.dss.stencilRefBack = stencilCompareMask.reference;
                            cache.dss.stencilReadMaskBack = stencilCompareMask.compareMask;
                        }
                        break;
                    }
                    }
                }
                break;
            }
            } // switch
        } // for
    } // update dynamic states
}

export function WebGPUCmdFuncDraw (device: WebGPUDevice, drawInfo: DrawInfo) {
    const gl = device.gl;
    const { gpuInputAssembler, glPrimitive } = gfxStateCache;

    if (gpuInputAssembler) {
        if (gpuInputAssembler.gpuIndirectBuffer) {
            const indirects = gpuInputAssembler.gpuIndirectBuffer.indirects;
            for (let k = 0; k < indirects.length; k++) {
                const subDrawInfo = indirects[k];
                const gpuBuffer = gpuInputAssembler.gpuIndexBuffer;
                if (subDrawInfo.instanceCount) {
                    if (gpuBuffer) {
                        if (subDrawInfo.indexCount > 0) {
                            const offset = subDrawInfo.firstIndex * gpuBuffer.stride;
                            gl.drawElementsInstanced(glPrimitive, subDrawInfo.indexCount,
                                gpuInputAssembler.glIndexType, offset, subDrawInfo.instanceCount);
                        }
                    } else if (subDrawInfo.vertexCount > 0) {
                        gl.drawArraysInstanced(glPrimitive, subDrawInfo.firstVertex, subDrawInfo.vertexCount, subDrawInfo.instanceCount);
                    }
                } else if (gpuBuffer) {
                    if (subDrawInfo.indexCount > 0) {
                        const offset = subDrawInfo.firstIndex * gpuBuffer.stride;
                        gl.drawElements(glPrimitive, subDrawInfo.indexCount, gpuInputAssembler.glIndexType, offset);
                    }
                } else if (subDrawInfo.vertexCount > 0) {
                    gl.drawArrays(glPrimitive, subDrawInfo.firstVertex, subDrawInfo.vertexCount);
                }
            }
        } else if (drawInfo.instanceCount) {
            if (gpuInputAssembler.gpuIndexBuffer) {
                if (drawInfo.indexCount > 0) {
                    const offset = drawInfo.firstIndex * gpuInputAssembler.gpuIndexBuffer.stride;
                    gl.drawElementsInstanced(glPrimitive, drawInfo.indexCount,
                        gpuInputAssembler.glIndexType, offset, drawInfo.instanceCount);
                }
            } else if (drawInfo.vertexCount > 0) {
                gl.drawArraysInstanced(glPrimitive, drawInfo.firstVertex, drawInfo.vertexCount, drawInfo.instanceCount);
            }
        } else if (gpuInputAssembler.gpuIndexBuffer) {
            if (drawInfo.indexCount > 0) {
                const offset = drawInfo.firstIndex * gpuInputAssembler.gpuIndexBuffer.stride;
                gl.drawElements(glPrimitive, drawInfo.indexCount, gpuInputAssembler.glIndexType, offset);
            }
        } else if (drawInfo.vertexCount > 0) {
            gl.drawArrays(glPrimitive, drawInfo.firstVertex, drawInfo.vertexCount);
        }
    }
}

const cmdIds = new Array<number>(WebGPUCmd.COUNT);
export function WebGPUCmdFuncExecuteCmds (device: WebGPUDevice, cmdPackage: WebGPUCmdPackage) {
    cmdIds.fill(0);

    for (let i = 0; i < cmdPackage.cmds.length; ++i) {
        const cmd = cmdPackage.cmds.array[i];
        const cmdId = cmdIds[cmd]++;

        switch (cmd) {
        case WebGPUCmd.BEGIN_RENDER_PASS: {
            const cmd0 = cmdPackage.beginRenderPassCmds.array[cmdId];
            WebGPUCmdFuncBeginRenderPass(device, cmd0.gpuRenderPass, cmd0.gpuFramebuffer, cmd0.renderArea,
                cmd0.clearColors, cmd0.clearDepth, cmd0.clearStencil);
            break;
        }
        /*
            case WebGPUCmd.END_RENDER_PASS: {
                // WebGL 2.0 doesn't support store operation of attachments.
                // GFXStoreOp.Store is the default GL behavior.
                break;
            }
            */
        case WebGPUCmd.BIND_STATES: {
            const cmd2 = cmdPackage.bindStatesCmds.array[cmdId];
            WebGPUCmdFuncBindStates(device, cmd2.gpuPipelineState, cmd2.gpuInputAssembler, cmd2.gpuDescriptorSets, cmd2.dynamicOffsets,
                cmd2.viewport, cmd2.scissor, cmd2.lineWidth, cmd2.depthBias, cmd2.blendConstants,
                cmd2.depthBounds, cmd2.stencilWriteMask, cmd2.stencilCompareMask);
            break;
        }
        case WebGPUCmd.DRAW: {
            const cmd3 = cmdPackage.drawCmds.array[cmdId];
            WebGPUCmdFuncDraw(device, cmd3.drawInfo);
            break;
        }
        case WebGPUCmd.UPDATE_BUFFER: {
            const cmd4 = cmdPackage.updateBufferCmds.array[cmdId];
            WebGPUCmdFuncUpdateBuffer(device, cmd4.gpuBuffer as IWebGPUGPUBuffer, cmd4.buffer as BufferSource, cmd4.offset, cmd4.size);
            break;
        }
        case WebGPUCmd.COPY_BUFFER_TO_TEXTURE: {
            const cmd5 = cmdPackage.copyBufferToTextureCmds.array[cmdId];
            WebGPUCmdFuncCopyBuffersToTexture(device, cmd5.buffers, cmd5.gpuTexture as IWebGPUGPUTexture, cmd5.regions);
            break;
        }
        } // switch
    } // for
}

function maxElementOfImageArray (bufInfoArr: BufferTextureCopy[]): number {
    let maxSize = 0;
    for (let i = 0; i < bufInfoArr.length; i++) {
        const curSize = bufInfoArr[i].texExtent.width * bufInfoArr[i].texExtent.height * bufInfoArr[i].texExtent.depth;
        maxSize = maxSize < curSize ? curSize : maxSize;
    }
    return maxSize;
}

/*
//rgba8unorm rgbafloat ...
type GLTypeManifest = { [key: string]: string | number;}
type WebGPUDataTypeManifest = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array | undefined;
type GLFormatTypeManifest<E extends GLTypeManifest> = { [key in E[keyof E]]:WebGPUDataTypeManifest };

//return type with | undefined may need to be encoded in shader manually
interface GFXDataTypeToRawArrayType extends GLFormatTypeManifest<typeof Format> {
    [Format.R8]: Uint8Array;
    [Format.R8SN]: Int8Array;
    [Format.R8UI]: Uint8Array;
    [Format.R8I]: Int8Array;
    [Format.R16UI]: Uint16Array;
    [Format.R16I]: Int16Array;
    [Format.R32F]: Float32Array;
    [Format.R32UI]: Uint32Array;
    [Format.R32I]: Int32Array;

    [Format.RG8]: Uint8Array;
    [Format.RG8SN]: Int8Array;
    [Format.RG8UI]: Uint8Array;
    [Format.RG8I]: Int8Array;
    [Format.RG16UI]: Uint16Array;
    [Format.RG16I]: Int16Array;
    [Format.RG32F]: Float32Array;
    [Format.RG32UI]: Uint32Array;
    [Format.RG32I]: Int32Array;

    [Format.RGB8]: Uint8Array;
    [Format.SRGB8]: Uint8Array;
    [Format.RGB8SN]: Int8Array;
    [Format.RGB8UI]: Uint8Array;
    [Format.RGB8I]: Int8Array;
    [Format.RGB16UI]: Uint16Array;
    [Format.RGB16I]: Int16Array;
    [Format.RGB32F]: Float32Array;
    [Format.RGB32UI]: Uint32Array;
    [Format.RGB32I]: Int32Array;

    [Format.BGRA8]: Uint8Array;
    [Format.RGBA8]: Int8Array;
    [Format.SRGB8_A8]: Uint8Array;
    [Format.RGBA8SN]: Int8Array;
    [Format.RGBA8UI]: Uint8Array;
    [Format.RGBA8I]: Int8Array;
    [Format.RGBA16UI]: Uint16Array;
    [Format.RGBA16I]: Int16Array;
    [Format.RGBA32F]: Float32Array;
    [Format.RGBA32UI]: Uint32Array;
    [Format.RGBA32I]: Int32Array;

    [Format.R5G6B5]: Uint16Array | undefined;
    [Format.R11G11B10F]: Uint32Array | undefined;
    [Format.RGB5A1]: Uint16Array | undefined;
    [Format.RGBA4]: Uint16Array | undefined;
    [Format.RGB10A2]: Uint32Array | undefined;
    [Format.RGB10A2UI]: Uint32Array | undefined;//wiki: GL_RGB10_A2UI no signed integral version.
    [Format.RGB9E5]: Float32Array | undefined;

    [Format.D16]: Uint16Array;
    [Format.D16S8]: Uint32Array | undefined;
    [Format.D24]: Uint32Array | undefined;
    [Format.D24S8]: Uint32Array | undefined;
    [Format.D32F]: Float32Array;

    [Format.BC1]: Uint8Array;
    [Format.BC1_SRGB]: Uint8Array;
    [Format.BC2]: Uint8Array;
    [Format.BC2_SRGB]: Uint8Array;
    [Format.BC3]: Uint8Array;
    [Format.BC3_SRGB]: Uint8Array;
    [Format.BC4]: Uint8Array;
    [Format.BC4_SNORM]: Int8Array;
    [Format.BC5]: Uint8Array;
    [Format.BC5_SNORM]: Int8Array;
    [Format.BC6H_SF16]: Float32Array;
    [Format.BC6H_UF16]: Float32Array;
    [Format.BC7]: Uint8Array;
    [Format.BC7_SRGB]: Uint8Array;
}
*/

// return type with | undefined may need to be encoded in shader manually
interface GFXDataTypeToRawArrayType {
    readonly [Format.R8]: Uint8Array;
    readonly [Format.R8SN]: Int8Array;
    readonly [Format.R8UI]: Uint8Array;
    readonly [Format.R8I]: Int8Array;
    readonly [Format.R16UI]: Uint16Array;
    readonly [Format.R16I]: Int16Array;
    readonly [Format.R32F]: Float32Array;
    readonly [Format.R32UI]: Uint32Array;
    readonly [Format.R32I]: Int32Array;

    readonly [Format.RG8]: Uint8Array;
    readonly [Format.RG8SN]: Int8Array;
    readonly [Format.RG8UI]: Uint8Array;
    readonly [Format.RG8I]: Int8Array;
    readonly [Format.RG16UI]: Uint16Array;
    readonly [Format.RG16I]: Int16Array;
    readonly [Format.RG32F]: Float32Array;
    readonly [Format.RG32UI]: Uint32Array;
    readonly [Format.RG32I]: Int32Array;

    readonly [Format.RGB8]: Uint8Array;
    readonly [Format.SRGB8]: Uint8Array;
    readonly [Format.RGB8SN]: Int8Array;
    readonly [Format.RGB8UI]: Uint8Array;
    readonly [Format.RGB8I]: Int8Array;
    readonly [Format.RGB16UI]: Uint16Array;
    readonly [Format.RGB16I]: Int16Array;
    readonly [Format.RGB32F]: Float32Array;
    readonly [Format.RGB32UI]: Uint32Array;
    readonly [Format.RGB32I]: Int32Array;

    readonly [Format.BGRA8]: Uint8Array;
    readonly [Format.RGBA8]: Int8Array;
    readonly [Format.SRGB8_A8]: Uint8Array;
    readonly [Format.RGBA8SN]: Int8Array;
    readonly [Format.RGBA8UI]: Uint8Array;
    readonly [Format.RGBA8I]: Int8Array;
    readonly [Format.RGBA16UI]: Uint16Array;
    readonly [Format.RGBA16I]: Int16Array;
    readonly [Format.RGBA32F]: Float32Array;
    readonly [Format.RGBA32UI]: Uint32Array;
    readonly [Format.RGBA32I]: Int32Array;

    readonly [Format.R5G6B5]: Uint16Array | undefined;
    readonly [Format.R11G11B10F]: Uint32Array | undefined;
    readonly [Format.RGB5A1]: Uint16Array | undefined;
    readonly [Format.RGBA4]: Uint16Array | undefined;
    readonly [Format.RGB10A2]: Uint32Array | undefined;
    readonly [Format.RGB10A2UI]: Uint32Array | undefined;// wiki: GL_RGB10_A2UI no signed integral version.
    readonly [Format.RGB9E5]: Float32Array | undefined;

    readonly [Format.D16]: Uint16Array;
    readonly [Format.D16S8]: Uint32Array | undefined;
    readonly [Format.D24]: Uint32Array | undefined;
    readonly [Format.D24S8]: Uint32Array | undefined;
    readonly [Format.D32F]: Float32Array;

    readonly [Format.BC1]: Uint8Array;
    readonly [Format.BC1_SRGB]: Uint8Array;
    readonly [Format.BC2]: Uint8Array;
    readonly [Format.BC2_SRGB]: Uint8Array;
    readonly [Format.BC3]: Uint8Array;
    readonly [Format.BC3_SRGB]: Uint8Array;
    readonly [Format.BC4]: Uint8Array;
    readonly [Format.BC4_SNORM]: Int8Array;
    readonly [Format.BC5]: Uint8Array;
    readonly [Format.BC5_SNORM]: Int8Array;
    readonly [Format.BC6H_SF16]: Float32Array;
    readonly [Format.BC6H_UF16]: Float32Array;
    readonly [Format.BC7]: Uint8Array;
    readonly [Format.BC7_SRGB]: Uint8Array;
}

export function WebGPUCmdFuncCopyTexImagesToTexture (
    device: WebGPUDevice,
    texImages: TexImageSource[],
    gpuTexture: IWebGPUGPUTexture,
    regions: BufferTextureCopy[],
) {
    // name all native webgpu resource nativeXXX distinguished from gpuTexture passed in.
    const nativeDevice = device.nativeDevice()!;

    // raw image buffer -> gpubuffer -> texture
    // all image share the same & largest buffer

    const pixelSize = FormatInfos[gpuTexture.format].size;
    const nativeBuffer = nativeDevice.createBuffer({
        size: maxElementOfImageArray(regions) * pixelSize,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    for (let i = 0; i < regions.length; i++) {
        const imageData = (texImages[i] as HTMLCanvasElement).getContext('2d')?.getImageData(0, 0, texImages[i].width, texImages[i].height);
        new Uint8ClampedArray(nativeBuffer.getMappedRange(
            0,
            imageData!.data.byteLength,
        )).set(imageData!.data);
        nativeBuffer.unmap();

        const commandEncoder = nativeDevice.createCommandEncoder();
        commandEncoder.copyBufferToTexture(
            { buffer: nativeBuffer, bytesPerRow: pixelSize },
            { texture: gpuTexture.glTexture! },
            [regions[i].texExtent.width, regions[i].texExtent.height, regions[i].texExtent.depth],
        );
        nativeDevice.defaultQueue.submit([commandEncoder.finish()]);
        // 👇async way in initial stage may cause issues
        // createImageBitmap(texImages[i]).then( (img:ImageBitmap) => {
        //    nativeDevice.defaultQueue.copyImageBitmapToTexture(
        //        {imageBitmap: img},
        //        {texture: gpuTexture.glTexture!},
        //        [regions[i].texExtent.width, regions[i].texExtent.height, regions[i].texExtent.depth]);
        // } );
    }

    nativeBuffer.destroy();

    // blitCommandEncoder.copyBufferToTexture(,)

    // device.nativeDevice().cop

    // const gl = device.gl;
    // const glTexUnit = device.stateCache.glTexUnits[device.stateCache.texUnit];
    // if (glTexUnit.glTexture !== gpuTexture.glTexture) {
    //     gl.bindTexture(gpuTexture.glTarget, gpuTexture.glTexture);
    //     glTexUnit.glTexture = gpuTexture.glTexture;
    // }

    // let n = 0;
    // let f = 0;

    // switch (gpuTexture.glTarget) {
    //     case gl.TEXTURE_2D: {
    //         for (let k = 0; k < regions.length; k++) {
    //             const region = regions[k];
    //             gl.texSubImage2D(gl.TEXTURE_2D, region.texSubres.mipLevel,
    //                 region.texOffset.x, region.texOffset.y,
    //                 gpuTexture.glFormat, gpuTexture.glType, texImages[n++]);
    //         }
    //         break;
    //     }
    //     case gl.TEXTURE_CUBE_MAP: {
    //         for (let k = 0; k < regions.length; k++) {
    //             const region = regions[k];
    //             const fcount = region.texSubres.baseArrayLayer + region.texSubres.layerCount;
    //             for (f = region.texSubres.baseArrayLayer; f < fcount; ++f) {
    //                 gl.texSubImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + f, region.texSubres.mipLevel,
    //                     region.texOffset.x, region.texOffset.y,
    //                     gpuTexture.glFormat, gpuTexture.glType, texImages[n++]);
    //             }
    //         }
    //         break;
    //     }
    //     default: {
    //         console.error('Unsupported GL texture type, copy buffer to texture failed.');
    //     }
    // }

    // if (gpuTexture.flags & TextureFlagBit.GEN_MIPMAP) {
    //     gl.generateMipmap(gpuTexture.glTarget);
    // }
}

export function WebGPUCmdFuncCopyBuffersToTexture (
    device: WebGPUDevice,
    buffers: ArrayBufferView[],
    gpuTexture: IWebGPUGPUTexture,
    regions: BufferTextureCopy[],
) {
    const nativeDevice = device.nativeDevice()!;

    for (let i = 0; i < regions.length; i++) {
        const texView = {} as GPUTextureCopyView;
        texView.texture = gpuTexture.glTexture!;
        const texDataLayout = {} as GPUTextureDataLayout;
        texDataLayout.bytesPerRow = FormatInfos[gpuTexture.format].size;
        nativeDevice.defaultQueue.writeTexture(texView, buffers[i], texDataLayout, regions[i].texExtent);
    }

    /*
    const gl = device.gl;
    const glTexUnit = device.stateCache.glTexUnits[device.stateCache.texUnit];
    if (glTexUnit.glTexture !== gpuTexture.glTexture) {
        gl.bindTexture(gpuTexture.glTarget, gpuTexture.glTexture);
        glTexUnit.glTexture = gpuTexture.glTexture;
    }

    let n = 0;
    let w = 1;
    let h = 1;
    let f = 0;
    const fmtInfo: FormatInfo = FormatInfos[gpuTexture.format];
    const isCompressed = fmtInfo.isCompressed;

    switch (gpuTexture.glTarget) {
        case gl.TEXTURE_2D: {
            for (let k = 0; k < regions.length; k++) {
                const region = regions[k];
                w = region.texExtent.width;
                h = region.texExtent.height;
                const pixels = buffers[n++];
                if (!isCompressed) {
                    gl.texSubImage2D(gl.TEXTURE_2D, region.texSubres.mipLevel,
                        region.texOffset.x, region.texOffset.y, w, h,
                        gpuTexture.glFormat, gpuTexture.glType, pixels);
                } else {
                    if (gpuTexture.glInternalFmt !== WebGLEXT.COMPRESSED_RGB_ETC1_WEBGL) {
                        gl.compressedTexSubImage2D(gl.TEXTURE_2D, region.texSubres.mipLevel,
                            region.texOffset.x, region.texOffset.y, w, h,
                            gpuTexture.glFormat, pixels);
                    } else {
                        gl.compressedTexImage2D(gl.TEXTURE_2D, region.texSubres.mipLevel,
                            gpuTexture.glInternalFmt, w, h, 0, pixels);
                    }
                }
            }
            break;
        }
        case gl.TEXTURE_CUBE_MAP: {
            for (let k = 0; k < regions.length; k++) {
                const region = regions[k];
                const fcount = region.texSubres.baseArrayLayer + region.texSubres.layerCount;
                for (f = region.texSubres.baseArrayLayer; f < fcount; ++f) {
                    w = region.texExtent.width;
                    h = region.texExtent.height;

                    const pixels = buffers[n++];

                    if (!isCompressed) {
                        gl.texSubImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + f, region.texSubres.mipLevel,
                            region.texOffset.x, region.texOffset.y, w, h,
                            gpuTexture.glFormat, gpuTexture.glType, pixels);
                    } else {
                        if (gpuTexture.glInternalFmt !== WebGLEXT.COMPRESSED_RGB_ETC1_WEBGL) {
                            gl.compressedTexSubImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + f, region.texSubres.mipLevel,
                                region.texOffset.x, region.texOffset.y, w, h,
                                gpuTexture.glFormat, pixels);
                        } else {
                            gl.compressedTexImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + f, region.texSubres.mipLevel,
                                gpuTexture.glInternalFmt, w, h, 0, pixels);
                        }
                    }
                }
            }
            break;
        }
        default: {
            console.error('Unsupported GL texture type, copy buffer to texture failed.');
        }
    }

    if (gpuTexture.flags & TextureFlagBit.GEN_MIPMAP) {
        gl.generateMipmap(gpuTexture.glTarget);
    }
    */
}

export function WebGPUCmdFuncBlitFramebuffer (
    device: WebGPUDevice,
    src: IWebGPUGPUFramebuffer,
    dst: IWebGPUGPUFramebuffer,
    srcRect: Rect,
    dstRect: Rect,
    filter: Filter,
) {
    const gl = device.gl;

    if (device.stateCache.glReadFramebuffer !== src.glFramebuffer) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, src.glFramebuffer);
        device.stateCache.glReadFramebuffer = src.glFramebuffer;
    }

    const rebindFBO = (dst.glFramebuffer !== device.stateCache.glFramebuffer);
    if (rebindFBO) {
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst.glFramebuffer);
    }

    let mask = 0;
    if (src.gpuColorTextures.length > 0) {
        mask |= gl.COLOR_BUFFER_BIT;
    }

    if (src.gpuDepthStencilTexture) {
        mask |= gl.DEPTH_BUFFER_BIT;
        if (FormatInfos[src.gpuDepthStencilTexture.format].hasStencil) {
            mask |= gl.STENCIL_BUFFER_BIT;
        }
    }

    const glFilter = (filter === Filter.LINEAR || filter === Filter.ANISOTROPIC) ? gl.LINEAR : gl.NEAREST;

    gl.blitFramebuffer(
        srcRect.x, srcRect.y, srcRect.x + srcRect.width, srcRect.y + srcRect.height,
        dstRect.x, dstRect.y, dstRect.x + dstRect.width, dstRect.y + dstRect.height,
        mask, glFilter,
    );

    if (rebindFBO) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, device.stateCache.glFramebuffer);
    }
}
