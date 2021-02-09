import glslang, { Glslang } from '@webgpu/glslang/dist/web-devel/glslang';
import { macro, warnID, warn } from '../../platform';
import { DescriptorSet, DescriptorSetInfo } from '../descriptor-set';
import { Buffer, BufferInfo, BufferViewInfo } from '../buffer';
import { CommandBuffer, CommandBufferInfo } from '../command-buffer';
import { Device, DeviceInfo, BindingMappingInfo } from '../device';
import { Fence, FenceInfo } from '../fence';
import { Framebuffer, FramebufferInfo } from '../framebuffer';
import { InputAssembler, InputAssemblerInfo } from '../input-assembler';
import { PipelineState, PipelineStateInfo } from '../pipeline-state';
import { Queue, QueueInfo } from '../queue';
import { RenderPass, RenderPassInfo } from '../render-pass';
import { Sampler, SamplerInfo } from '../sampler';
import { Shader, ShaderInfo } from '../shader';
import { PipelineLayout, PipelineLayoutInfo } from '../pipeline-layout';
import { DescriptorSetLayout, DescriptorSetLayoutInfo } from '../descriptor-set-layout';
import { Texture, TextureInfo, TextureViewInfo } from '../texture';
import { WebGPUDescriptorSet } from './webgpu-descriptor-set';
import { WebGPUBuffer } from './webgpu-buffer';
import { WebGPUCommandBuffer } from './webgpu-command-buffer';
import { WebGPUFence } from './webgpu-fence';
import { WebGPUFramebuffer } from './webgpu-framebuffer';
import { WebGPUInputAssembler } from './webgpu-input-assembler';
import { WebGPUDescriptorSetLayout } from './webgpu-descriptor-set-layout';
import { WebGPUPipelineLayout } from './webgpu-pipeline-layout';
import { WebGPUPipelineState } from './webgpu-pipeline-state';
import { WebGPUQueue } from './webgpu-queue';
import { WebGPURenderPass } from './webgpu-render-pass';
import { WebGPUSampler } from './webgpu-sampler';
import { WebGPUShader } from './webgpu-shader';
import { WebGPUStateCache } from './webgpu-state-cache';
import { WebGPUTexture } from './webgpu-texture';
import { GFXFormatToWebGLFormat, GFXFormatToWebGLType, WebGPUCmdFuncBlitFramebuffer,
    WebGPUCmdFuncCopyBuffersToTexture, WebGPUCmdFuncCopyTexImagesToTexture } from './webgpu-commands';
import { getTypedArrayConstructor, CommandBufferType, Filter, Format, FormatInfos,
    QueueType, TextureFlagBit, TextureType, TextureUsageBit,  API, Feature } from '../define';
import { BufferTextureCopy, Rect } from '../define-class';
import { WebGPUCommandAllocator } from './webgpu-command-allocator';
// import { Glslang } from './glslang';
// import getGlslang from './glslang';

// namespace glslangModule {
//     export type ShaderStage = 'vertex' | 'fragment' | 'compute';
//     export type SpirvVersion = '1.0' | '1.1' | '1.2' | '1.3' | '1.4' | '1.5';

//     export interface ResultZeroCopy {
//         readonly data: Uint32Array;
//         free (): void;
//     }

//     export interface Glslang {
//         compileGLSLZeroCopy (glsl: string, shader_stage: ShaderStage, gen_debug: boolean, spirv_version?: SpirvVersion): ResultZeroCopy;
//         compileGLSL (glsl: string, shader_type: ShaderStage, gen_debug: boolean, spirv_version?: SpirvVersion): Uint32Array;
//     }
// }

export class WebGPUDevice extends Device {
    get gl (): WebGL2RenderingContext {
        return null!;
    }

    get isAntialias (): boolean {
        return null!;
    }

    get isPremultipliedAlpha (): boolean {
        return null!;
    }

    get useVAO (): boolean {
        return null!;
    }

    get bindingMappingInfo () {
        return this._bindingMappingInfo;
    }

    get EXT_texture_filter_anisotropic () {
        return null!;
    }

    get OES_texture_float_linear () {
        return null!;
    }

    get EXT_color_buffer_float () {
        return null!;
    }

    get EXT_disjoint_timer_query_webgl2 () {
        return null!;
    }

    get WEBGL_compressed_texture_etc1 () {
        return null!;
    }

    get WEBGL_compressed_texture_etc () {
        return null!;
    }

    get WEBGL_compressed_texture_pvrtc () {
        return null!;
    }

    get WEBGL_compressed_texture_s3tc () {
        return null!;
    }

    get WEBGL_compressed_texture_s3tc_srgb () {
        return null!;
    }

    get WEBGL_texture_storage_multisample () {
        return null!;
    }

    get WEBGL_debug_shaders () {
        return null!;
    }

    get WEBGL_lose_context () {
        return null!;
    }

    get indexedIndirect () {
        return this._indexedIndirect;
    }

    public stateCache: WebGPUStateCache = new WebGPUStateCache();
    public cmdAllocator: WebGPUCommandAllocator = new WebGPUCommandAllocator();
    public nullTex2D: WebGPUTexture | null = null;
    public nullTexCube: WebGPUTexture | null = null;

    private _adapter: GPUAdapter | null | undefined = null;
    private _device: GPUDevice | null | undefined = null;
    private _context: GPUCanvasContext | null = null;
    private _swapChain: GPUSwapChain | null = null;
    private _glslang: Glslang | null = null;
    private _bindingMappingInfo: BindingMappingInfo = new BindingMappingInfo();
    private _indexedIndirect = false;

    public initialize (info: DeviceInfo): Promise<boolean> {
        return this.initDevice(info);
    }

    private async initDevice (info: DeviceInfo): Promise<boolean> {
        this._bindingMappingInfo = info.bindingMappingInfo;
        if (!this._bindingMappingInfo.bufferOffsets.length) this._bindingMappingInfo.bufferOffsets.push(0);
        if (!this._bindingMappingInfo.samplerOffsets.length) this._bindingMappingInfo.samplerOffsets.push(0);

        this._adapter = await navigator.gpu?.requestAdapter();
        this._device = await this._adapter?.requestDevice();
        this._glslang = await glslang();

        this._canvas = info.canvasElm as HTMLCanvasElement;
        this._context = this._canvas.getContext('gpupresent')!;
        const swapchainFormat = 'bgra8unorm';

        const device: GPUDevice = this._device as GPUDevice;
        this._swapChain = this._context.configureSwapChain({
            device,
            format: swapchainFormat,
        });

        // FIXME: require by query
        this._indexedIndirect = false;

        this._queue = this.createQueue(new QueueInfo(QueueType.GRAPHICS));
        this._cmdBuff = this.createCommandBuffer(new CommandBufferInfo(this._queue));

        // glslangModule().then((glslang) => {
        //     this._glslang = glslang;
        // });
        return true;
    }

    public destroy (): void {
    }

    public resize (width: number, height: number) {
    }

    public acquire () {
    }

    get swapChain () {
        return this._swapChain;
    }

    public nativeDevice () {
        return this._device;
    }

    public glslang () {
        return this._glslang;
    }

    public present () {
        const queue = (this._queue as WebGPUQueue);
        this._numDrawCalls = queue.numDrawCalls;
        this._numInstances = queue.numInstances;
        this._numTris = queue.numTris;
        queue.clear();
    }

    public createCommandBuffer (info: CommandBufferInfo): CommandBuffer {
        const cmdBuff = new WebGPUCommandBuffer(this);
        if (cmdBuff.initialize(info)) {
            return cmdBuff;
        }
        return null!;
    }

    public createBuffer (info: BufferInfo | BufferViewInfo): Buffer {
        const buffer = new WebGPUBuffer(this);
        if (buffer.initialize(info)) {
            return buffer;
        }
        return null!;
    }

    public createTexture (info: TextureInfo | TextureViewInfo): Texture {
        const texture = new WebGPUTexture(this);
        if (texture.initialize(info)) {
            return texture;
        }
        return null!;
    }

    public createSampler (info: SamplerInfo): Sampler {
        const sampler = new WebGPUSampler(this);
        if (sampler.initialize(info)) {
            return sampler;
        }
        return null!;
    }

    public createDescriptorSet (info: DescriptorSetInfo): DescriptorSet {
        const descriptorSet = new WebGPUDescriptorSet(this);
        if (descriptorSet.initialize(info)) {
            return descriptorSet;
        }
        return null!;
    }

    public createShader (info: ShaderInfo): Shader {
        const shader = new WebGPUShader(this);
        if (shader.initialize(info)) {
            return shader;
        }
        return null!;
    }

    public createInputAssembler (info: InputAssemblerInfo): InputAssembler {
        const inputAssembler = new WebGPUInputAssembler(this);
        if (inputAssembler.initialize(info)) {
            return inputAssembler;
        }
        return null!;
    }

    public createRenderPass (info: RenderPassInfo): RenderPass {
        const renderPass = new WebGPURenderPass(this);
        if (renderPass.initialize(info)) {
            return renderPass;
        }
        return null!;
    }

    public createFramebuffer (info: FramebufferInfo): Framebuffer {
        const framebuffer = new WebGPUFramebuffer(this);
        if (framebuffer.initialize(info)) {
            return framebuffer;
        }
        return null!;
    }

    public createDescriptorSetLayout (info: DescriptorSetLayoutInfo): DescriptorSetLayout {
        const descriptorSetLayout = new WebGPUDescriptorSetLayout(this);
        if (descriptorSetLayout.initialize(info)) {
            return descriptorSetLayout;
        }
        return null!;
    }

    public createPipelineLayout (info: PipelineLayoutInfo): PipelineLayout {
        const pipelineLayout = new WebGPUPipelineLayout(this);
        if (pipelineLayout.initialize(info)) {
            return pipelineLayout;
        }
        return null!;
    }

    public createPipelineState (info: PipelineStateInfo): PipelineState {
        const pipelineState = new WebGPUPipelineState(this);
        if (pipelineState.initialize(info)) {
            return pipelineState;
        }
        return null!;
    }

    public createFence (info: FenceInfo): Fence {
        const fence = new WebGPUFence(this);
        if (fence.initialize(info)) {
            return fence;
        }
        return null!;
    }

    public createQueue (info: QueueInfo): Queue {
        const queue = new WebGPUQueue(this);
        if (queue.initialize(info)) {
            return queue;
        }
        return null!;
    }

    public copyBuffersToTexture (buffers: ArrayBufferView[], texture: Texture, regions: BufferTextureCopy[]) {
        WebGPUCmdFuncCopyBuffersToTexture(
            this,
            buffers,
            (texture as WebGPUTexture).gpuTexture,
            regions,
        );
    }

    public copyTexImagesToTexture (
        texImages: TexImageSource[],
        texture: Texture,
        regions: BufferTextureCopy[],
    ) {
        WebGPUCmdFuncCopyTexImagesToTexture(
            this,
            texImages,
            (texture as WebGPUTexture).gpuTexture,
            regions,
        );
    }

    public copyFramebufferToBuffer (
        srcFramebuffer: Framebuffer,
        dstBuffer: ArrayBuffer,
        regions: BufferTextureCopy[],
    ) {}

    public blitFramebuffer (src: Framebuffer, dst: Framebuffer, srcRect: Rect, dstRect: Rect, filter: Filter) {}
}
