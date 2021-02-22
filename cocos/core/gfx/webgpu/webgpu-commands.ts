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
import { math } from '../..';

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
    case DescriptorType.DYNAMIC_UNIFORM_BUFFER:
        return 'uniform-buffer';
    case DescriptorType.STORAGE_BUFFER:
        return 'storage-buffer';
    case DescriptorType.SAMPLER:
        return 'sampler';
    default:
        console.error('binding type not support by webGPU!');
    }
}

export function GFXFormatToWGPUVertexFormat (format: Format) :GPUVertexFormat {
    switch (format) {
    case Format.R32F: return 'float';
    case Format.R32UI: return 'uint';
    case Format.R32I: return 'int';

    case Format.RG8: return 'uchar2norm';
    case Format.RG8SN: return 'char2norm';
    case Format.RG8UI: return 'uchar2';
    case Format.RG8I: return 'char2';
    case Format.RG16F: return 'half2';
    case Format.RG16UI: return 'ushort2';
    case Format.RG16I: return 'short2';
    case Format.RG32F: return 'float2';
    case Format.RG32UI: return 'uint2';
    case Format.RG32I: return 'int2';

    case Format.RGB32F: return 'float3';
    case Format.RGB32UI: return 'uint3';
    case Format.RGB32I: return 'int3';

    case Format.BGRA8: return 'uchar4norm';
    case Format.RGBA8: return 'uchar4norm';
    case Format.SRGB8_A8: return 'char4';
    case Format.RGBA8SN: return 'char4norm';
    case Format.RGBA8UI: return 'uchar4';
    case Format.RGBA8I: return 'char4';
    case Format.RGBA16F: return 'half4';
    case Format.RGBA16UI: return 'ushort4';
    case Format.RGBA16I: return 'short4';
    case Format.RGBA32F: return 'float4';
    case Format.RGBA32UI: return 'uint4';
    case Format.RGBA32I: return 'int4';

    default: {
        console.warn('unexpected format for WGPU detected, return \'int\' as default.');
        return 'int';
    }
    }
}

export function GFXFormatToWGPUTextureFormat (format: Format): GPUTextureFormat  {
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
        console.info('Unsupported Format, convert to WebGPU internal format failed.');
        return 'rgba8unorm';
    }
    }
}

export function GFXFormatToWGPUFormat (format: Format): GPUTextureFormat {
    return GFXFormatToWGPUTextureFormat(format);
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
        nativeUsage |= GPUTextureUsage.RENDER_ATTACHMENT;
    }

    if (TextureUsageBit.DEPTH_STENCIL_ATTACHMENT) {
        nativeUsage |= GPUTextureUsage.RENDER_ATTACHMENT;
    }

    if (typeof nativeUsage === 'undefined') {
        console.error('Unsupported texture usage, convert to webGPU type failed.');
        nativeUsage = GPUTextureUsage.RENDER_ATTACHMENT;
    }

    return nativeUsage;
}

function GFXTypeToWGPUType (type: Type, gl: WebGL2RenderingContext): GLenum {
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
        console.info('Unsupported GFXBufferType yet, create UNIFORM buffer in default.');
        bufferUsage |= GPUBufferUsage.UNIFORM;
    }

    // FIXME: depend by inputs but vertexbuffer was updated without COPY_DST.
    bufferUsage |= GPUBufferUsage.COPY_DST;

    bufferDesc.usage = bufferUsage;
    gpuBuffer.glTarget = bufferUsage;
    gpuBuffer.glBuffer = nativeDevice.createBuffer(bufferDesc);
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
        nativeDevice.queue.submit([commandBuffer]);
        stagingBuffer.destroy();
    }
}

export function WebGPUCmdFuncCreateTexture (device: WebGPUDevice, gpuTexture: IWebGPUGPUTexture) {
    // dimension optional
    // let dim: GPUTextureViewDimension = GFXTextureToWebGPUTexture(gpuTexture.type);

    gpuTexture.glTarget = GFXTextureToWebGPUTexture(gpuTexture.type) as GPUTextureDimension;
    gpuTexture.glInternalFmt = GFXFormatToWGPUTextureFormat(gpuTexture.format);
    gpuTexture.glFormat = GFXFormatToWGPUFormat(gpuTexture.format);
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

function removeCombinedSamplerTexture (shaderSource: string) {
    const samplerTexturArr = shaderSource.match(/layout\(set = \d+, binding = \d+\) uniform sampler\w* \w+;/g);
    const count = samplerTexturArr?.length ? samplerTexturArr?.length : 0;

    let code = shaderSource;
    samplerTexturArr?.every((str) => {
        const textureName = str.match(/(?<=uniform sampler\w* )(\w+)(?=;)/g)!.toString();
        let samplerStr = str.replace(textureName, `${textureName}Sampler`);
        const samplerFunc = samplerStr.match(/(?<=uniform )(sampler\w*)/g)?.toString();
        samplerStr = samplerStr.replace(/(?<=uniform )(sampler\w*)/g, 'sampler');

        const textureReg = /(?<=binding = )(\d+)(?=\))/g;
        const textureBindingStr = str.match(textureReg)!.toString();
        const textureBinding = Number(textureBindingStr) + 20;
        let textureStr = str.replace(textureReg, textureBinding.toString());
        textureStr = textureStr.replace(/(?<=uniform )(sampler)(?=\w*)/g, 'texture');

        code = code.replace(str, `${samplerStr}\n${textureStr}`);

        const regStr = `texture(${textureName}`;
        while (code.indexOf(regStr) !== -1) {
            code = code.replace(regStr, `texture(${samplerFunc!}(${textureName}, ${textureName}Sampler)`);
        }

        return true;
    });
    return code;
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
        let sourceCode = `#version 450\n${gpuStage.source}`;
        sourceCode = removeCombinedSamplerTexture(sourceCode);
        const code = useWGSL ? sourceCode : glslang.compileGLSL(sourceCode, shaderTypeStr, true);
        const shader: GPUShaderModule = nativeDevice?.createShaderModule({ code });
        // shader.compilationInfo().then((compileInfo: GPUCompilationInfo) => {
        //     compileInfo.messages.forEach((info) => {
        //         console.log(info.lineNum, info.linePos, info.type, info.message);
        //     });
        // }).catch((compileInfo: GPUCompilationInfo) => {
        //     compileInfo.messages.forEach((info) => {
        //         console.log(info.lineNum, info.linePos, info.type, info.message);
        //     });
        // });
        const shaderStage: GPUProgrammableStageDescriptor = {
            module: shader,
            entryPoint: 'main',
        };
        gpuStage.glShader = shaderStage;
        // const complieInfo = shader.compilationInfo();
        // void complieInfo.then((info) => {
        //     console.info(info);
        // });
    }
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
    // const nativeBuffer = nativeDevice.createBuffer({
    //     size: maxElementOfImageArray(regions) * pixelSize,
    //     usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    //     mappedAtCreation: true,
    // });

    for (let i = 0; i < regions.length; i++) {
        const imageData = (texImages[i] as HTMLCanvasElement).getContext('2d')?.getImageData(0, 0, texImages[i].width, texImages[i].height);
        // new Uint8ClampedArray(nativeBuffer.getMappedRange(
        //     0,
        //     imageData!.data.byteLength,
        // )).set(imageData!.data);
        // nativeBuffer.unmap();

        const texDataLayout = {
            bytesPerRow: pixelSize * texImages[i].width,
        };
        const textureView:GPUTextureCopyView = {
            texture: gpuTexture.glTexture!,
        };
        nativeDevice.queue.writeTexture(textureView,
            imageData?.data.buffer as ArrayBuffer, texDataLayout,
            [regions[i].texExtent.width, regions[i].texExtent.height, regions[i].texExtent.depth]);
        // const commandEncoder = nativeDevice.createCommandEncoder();
        // commandEncoder.copyBufferToTexture(
        //     { buffer: nativeBuffer, bytesPerRow: Math.floor((pixelSize + 255) / 256) * 256 },
        //     { texture: gpuTexture.glTexture! },
        //     [regions[i].texExtent.width, regions[i].texExtent.height, regions[i].texExtent.depth],
        // );

        // const cmdBuffer = commandEncoder.finish();
        // nativeDevice.queue.submit([cmdBuffer]);
        // 👇async way in initial stage may cause issues
        // createImageBitmap(texImages[i]).then( (img:ImageBitmap) => {
        //    nativeDevice.queue.copyImageBitmapToTexture(
        //        {imageBitmap: img},
        //        {texture: gpuTexture.glTexture!},
        //        [regions[i].texExtent.width, regions[i].texExtent.height, regions[i].texExtent.depth]);
        // } );
    }

    // nativeBuffer.destroy();

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
        texDataLayout.bytesPerRow = FormatInfos[gpuTexture.format].size * regions[i].texExtent.width;
        nativeDevice.queue.writeTexture(texView, buffers[i],
            texDataLayout,
            {
                width: regions[i].texExtent.width,
                height: regions[i].texExtent.height,
                depthOrArrayLayers: regions[i].texExtent.depth,
            });
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
