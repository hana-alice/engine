import { DrawInfo } from '../buffer';
import {
    Address,
    DescriptorType,
    BufferUsage,
    Filter,
    Format,
    MemoryUsage,
    SampleCount,
    ShaderStageFlagBit,
    TextureFlags,
    TextureType,
    TextureUsage,
    Type,
    DynamicStateFlagBit,
} from '../define';
import { Attribute } from '../input-assembler';
import { BlendState, DepthStencilState, RasterizerState } from '../pipeline-state';
import { ColorAttachment, DepthStencilAttachment } from '../render-pass';
import { UniformBlock, UniformSampler } from '../shader';
import { DescriptorSetLayout, DescriptorSetLayoutBinding } from '../descriptor-set-layout';
import { WebGPUTexture } from './webgpu-texture';
import { WebGPUSampler } from './webgpu-sampler';

export interface IWebGPUGPUUniformInfo {
    name: string;
    type: Type;
    count: number;
    offset: number;
    view: Float32Array | Int32Array;
    isDirty: boolean;
}

export interface IWebGPUGPUBuffer {
    usage: BufferUsage;
    memUsage: MemoryUsage;
    size: number;
    stride: number;

    glTarget: GLenum;
    glBuffer: GPUBuffer | null;
    glOffset: number;

    buffer: ArrayBufferView | null;
    indirects: DrawInfo[];
    drawIndirectByIndex: boolean;
}

export interface IWebGPUGPUTexture {
    type: TextureType;
    format: Format;
    usage: TextureUsage;
    width: number;
    height: number;
    depth: number;
    size: number;
    arrayLayer: number;
    mipLevel: number;
    samples: SampleCount;
    flags: TextureFlags;
    isPowerOf2: boolean;

    glTarget: GPUTextureDimension;  // 1d, 2d, 3d
    glInternalFmt: GPUTextureFormat;// rgba8unorm
    glFormat: GPUTextureFormat;
    glType: GLenum;                 // data type, gl.UNSIGNED_BYTE
    glUsage: GPUTextureUsageFlags;  // webgl:DYNIMIC_DRAW... -> webGPU:COPY_DST/STORAGE...
    glTexture: GPUTexture | null;   // native tex handler
    glRenderbuffer: null;           // not suitable for webgpu
    glWrapS: GPUAddressMode;        // clamp-to-edge, repeat...
    glWrapT: GPUAddressMode;
    glMinFilter: GPUFilterMode;     // linear, nearest
    glMagFilter: GPUFilterMode;
}

export interface IWebGPUGPURenderPass {
    colorAttachments: ColorAttachment[];
    depthStencilAttachment: DepthStencilAttachment | null;
    nativeRenderPass: GPURenderPassDescriptor | null;
}

export interface IWebGPUGPUFramebuffer {
    gpuRenderPass: IWebGPUGPURenderPass;
    gpuColorTextures: IWebGPUGPUTexture[];
    gpuDepthStencilTexture: IWebGPUGPUTexture | null;
    isOffscreen?: boolean;

    glFramebuffer: WebGLFramebuffer | null;
}

export interface IWebGPUGPUSampler {
    glSampler: GPUSampler | null;
    minFilter: Filter;
    magFilter: Filter;
    mipFilter: Filter;
    addressU: Address;
    addressV: Address;
    addressW: Address;
    minLOD: number;
    maxLOD: number;

    glMinFilter: GPUFilterMode;
    glMagFilter: GPUFilterMode;
    glMipFilter: GPUFilterMode;
    glWrapS: GPUAddressMode;
    glWrapT: GPUAddressMode;
    glWrapR: GPUAddressMode;
}

export interface IWebGPUGPUInput {
    name: string;
    type: Type;
    stride: number;
    count: number;
    size: number;

    glType: GLenum;
    glLoc: GLint;
}

export interface IWebGPUGPUUniform {
    binding: number;
    name: string;
    type: Type;
    stride: number;
    count: number;
    size: number;
    offset: number;

    glType: GLenum;
    glLoc: WebGLUniformLocation;
    array: number[];
    begin: number;
}

export interface IWebGPUGPUUniformBlock {
    set: number;
    binding: number;
    idx: number;
    name: string;
    size: number;
    glBinding: number;
}

export interface IWebGPUGPUUniformSampler {
    set: number;
    binding: number;
    name: string;
    type: Type;
    count: number;
    units: number[];
    glUnits: Int32Array;

    glType: GLenum;
    glLoc: WebGLUniformLocation;
}

export interface IWebGPUGPUShaderStage {
    type: ShaderStageFlagBit;
    source: string;
    glShader: GPUProgrammableStageDescriptor | null;
}

export interface IWebGPUGPUShader {
    name: string;
    blocks: UniformBlock[];
    samplers: UniformSampler[];

    gpuStages: IWebGPUGPUShaderStage[];
    glProgram: WebGLProgram | null;
    glInputs: IWebGPUGPUInput[];
    glUniforms: IWebGPUGPUUniform[];
    glBlocks: IWebGPUGPUUniformBlock[];
    glSamplers: IWebGPUGPUUniformSampler[];
}

export interface IWebGPUGPUDescriptorSetLayout {
    bindings: DescriptorSetLayoutBinding[];
    dynamicBindings: number[];
    descriptorIndices: number[];
    descriptorCount: number;
    bindGroupLayout: GPUBindGroupLayout;
}

export interface IWebGPUGPUPipelineLayout {
    gpuSetLayouts: IWebGPUGPUDescriptorSetLayout[];
    dynamicOffsetCount: number;
    dynamicOffsetIndices: number[][];
    nativePipelineLayout: GPUPipelineLayout;
}

export interface IWebGPUGPUPipelineState {
    glPrimitive: GPUPrimitiveTopology;
    gpuShader: IWebGPUGPUShader | null;
    gpuPipelineLayout: IWebGPUGPUPipelineLayout | null;
    rs: RasterizerState;
    dss: DepthStencilState;
    bs: BlendState;
    dynamicStates: DynamicStateFlagBit[];
    gpuRenderPass: IWebGPUGPURenderPass | null;
    nativePipeline: GPUPipelineBase | undefined;
}

export interface IWebGPUGPUDescriptor {
    type: DescriptorType;
    gpuBuffer: IWebGPUGPUBuffer | null;
    gpuTexture: IWebGPUGPUTexture | null;
    gpuSampler: IWebGPUGPUSampler | null;
}

export interface IWebGPUGPUDescriptorSet {
    gpuDescriptors: IWebGPUGPUDescriptor[];
    descriptorIndices: number[];
    bindGroup: GPUBindGroup;
}

export interface IWebGPUAttrib {
    name: string;
    glBuffer: GPUBuffer | null;
    glType: GLenum;
    size: number;
    count: number;
    stride: number;
    componentCount: number;
    isNormalized: boolean;
    isInstanced: boolean;
    offset: number;
}

export interface IWebGPUGPUInputAssembler {
    attributes: Attribute[];
    gpuVertexBuffers: IWebGPUGPUBuffer[];
    gpuIndexBuffer: IWebGPUGPUBuffer | null;
    gpuIndirectBuffer: IWebGPUGPUBuffer | null;

    glAttribs: IWebGPUAttrib[];
    glIndexType: GPUIndexFormat;
    glVAOs: Map<WebGLProgram, WebGLVertexArrayObject>;
}
