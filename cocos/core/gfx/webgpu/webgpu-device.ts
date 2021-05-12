import glslang, { Glslang } from '@webgpu/glslang/dist/web-devel/glslang';
import { DescriptorSet, DescriptorSetResource } from '../base/descriptor-set';
import { Buffer } from '../base/buffer';
import { CommandBuffer } from '../base/command-buffer';
import { Device } from '../base/device';
import { Framebuffer } from '../base/framebuffer';
import { InputAssembler } from '../base/input-assembler';
import { PipelineState, PipelineStateInfo } from '../base/pipeline-state';
import { Queue } from '../base/queue';
import { RenderPass } from '../base/render-pass';
import { Sampler } from '../base/sampler';
import { Shader } from '../base/shader';
import { PipelineLayout } from '../base/pipeline-layout';
import { DescriptorSetLayout } from '../base/descriptor-set-layout';
import { Texture } from '../base/texture';
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
import { WebGPUCmdFuncCopyBuffersToTexture, WebGPUCmdFuncCopyTexImagesToTexture } from './webgpu-commands';
import {
    Filter, Format,
    QueueType, TextureFlagBit, TextureType, TextureUsageBit, Feature, SampleCount,
    BufferUsageBit, MemoryUsageBit, BufferFlagBit, BufferTextureCopy, Rect, DescriptorSetInfo,
    BufferInfo, BufferViewInfo, CommandBufferInfo, DeviceInfo, BindingMappingInfo,
    FramebufferInfo, InputAssemblerInfo, QueueInfo, RenderPassInfo, SamplerInfo,
    ShaderInfo, PipelineLayoutInfo, DescriptorSetLayoutInfo, TextureInfo, TextureViewInfo, GlobalBarrierInfo, TextureBarrierInfo
} from '../base/define';
import { WebGPUCommandAllocator } from './webgpu-command-allocator';
import { GlobalBarrier } from '../base/global-barrier';
import { TextureBarrier } from '../base/texture-barrier';

export class WebGPUDevice extends Device {
    public flushCommands(cmdBuffs: CommandBuffer[]): void {
        throw new Error('Method not implemented.');
    }
    public createGlobalBarrier(info: GlobalBarrierInfo): GlobalBarrier {
        throw new Error('Method not implemented.');
    }
    public createTextureBarrier(info: TextureBarrierInfo): TextureBarrier {
        throw new Error('Method not implemented.');
    }
    get gl(): WebGL2RenderingContext {
        return null!;
    }

    get isAntialias(): boolean {
        return null!;
    }

    get isPremultipliedAlpha(): boolean {
        return null!;
    }

    get useVAO(): boolean {
        return null!;
    }

    get bindingMappingInfo() {
        return this._bindingMappingInfo;
    }

    get EXT_texture_filter_anisotropic() {
        return null!;
    }

    get OES_texture_float_linear() {
        return null!;
    }

    get EXT_color_buffer_float() {
        return null!;
    }

    get EXT_disjoint_timer_query_webgl2() {
        return null!;
    }

    get WEBGL_compressed_texture_etc1() {
        return null!;
    }

    get WEBGL_compressed_texture_etc() {
        return null!;
    }

    get WEBGL_compressed_texture_pvrtc() {
        return null!;
    }

    get WEBGL_compressed_texture_s3tc() {
        return null!;
    }

    get WEBGL_compressed_texture_s3tc_srgb() {
        return null!;
    }

    get WEBGL_texture_storage_multisample() {
        return null!;
    }

    get WEBGL_debug_shaders() {
        return null!;
    }

    get WEBGL_lose_context() {
        return null!;
    }

    get multiDrawIndirectSupport() {
        return this._multiDrawIndirect;
    }

    get defaultColorTex() {
        return this._swapChain?.getCurrentTexture();
    }

    get defaultDepthStencilTex() {
        return this._defaultDepthStencilTex;
    }

    public stateCache: WebGPUStateCache = new WebGPUStateCache();
    public cmdAllocator: WebGPUCommandAllocator = new WebGPUCommandAllocator();
    public nullTex2D: WebGPUTexture | null = null;
    public nullTexCube: WebGPUTexture | null = null;
    public defaultDescriptorResource: DescriptorSetResource | null = null;

    private _adapter: GPUAdapter | null | undefined = null;
    private _device: GPUDevice | null | undefined = null;
    private _context: GPUCanvasContext | null = null;
    private _swapChain: GPUSwapChain | null = null;
    private _glslang: Glslang | null = null;
    private _bindingMappingInfo: BindingMappingInfo = new BindingMappingInfo();
    private _multiDrawIndirect = false;
    private _defaultDepthStencilTex: GPUTexture | null = null;

    public initialize(info: DeviceInfo): Promise<boolean> {
        return this.initDevice(info);
    }

    private async initDevice(info: DeviceInfo): Promise<boolean> {
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

        this._width = this._canvas.width * info.devicePixelRatio;
        this._height = this._canvas.height * info.devicePixelRatio;

        this._defaultDepthStencilTex = device.createTexture({
            size: {
                width: this._width,
                height: this.height,
                depthOrArrayLayers: 1,
            },
            format: 'depth24plus-stencil8',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this._devicePixelRatio = info.devicePixelRatio || 1.0;
        this._width = this._canvas.width;
        this._height = this._canvas.height;
        this._nativeWidth = Math.max(info.nativeWidth || this._width, 0);
        this._nativeHeight = Math.max(info.nativeHeight || this._height, 0);
        this._bindingMappingInfo = info.bindingMappingInfo;
        this._colorFmt = Format.BGRA8;
        this._caps.clipSpaceMinZ = 0.0;
        this._caps.uboOffsetAlignment = 256;

        // FIXME: require by query
        this._multiDrawIndirect = false;

        this._features.fill(false);
        this._features[Feature.TEXTURE_FLOAT] = true;
        this._features[Feature.TEXTURE_HALF_FLOAT] = true;
        this._features[Feature.FORMAT_RGB8] = true;
        this._features[Feature.FORMAT_D32F] = true;
        this._features[Feature.FORMAT_D24S8] = true;
        this._features[Feature.MSAA] = true;
        this._features[Feature.ELEMENT_INDEX_UINT] = true;
        this._features[Feature.INSTANCED_ARRAYS] = true;
        this._features[Feature.MULTIPLE_RENDER_TARGETS] = true;
        this._features[Feature.BLEND_MINMAX] = true;

        this._queue = this.createQueue(new QueueInfo(QueueType.GRAPHICS));
        this._cmdBuff = this.createCommandBuffer(new CommandBufferInfo(this._queue));

        const texInfo = new TextureInfo(
            TextureType.TEX2D,
            TextureUsageBit.NONE,
            Format.RGBA8,
            16,
            16,
            TextureFlagBit.NONE,
            1,
            1,
            SampleCount.X1,
            1,
        );
        const defaultDescTexResc = this.createTexture(texInfo);

        const bufferInfo = new BufferInfo(
            BufferUsageBit.NONE,
            MemoryUsageBit.NONE,
            16,
            16, // in bytes
            BufferFlagBit.NONE,
        );
        const defaultDescBuffResc = this.createBuffer(bufferInfo);

        const samplerInfo = new SamplerInfo();
        const defaultDescSmplResc = this.createSampler(samplerInfo);

        this.defaultDescriptorResource = {
            buffer: defaultDescBuffResc,
            texture: defaultDescTexResc,
            sampler: defaultDescSmplResc,
        };

        return true;
    }

    public destroy(): void {
        if (this._defaultDepthStencilTex) {
            this._defaultDepthStencilTex?.destroy();
        }

        if (this.defaultDescriptorResource) {
            if (this.defaultDescriptorResource.buffer) {
                this.defaultDescriptorResource.buffer.destroy();
            }
            if (this.defaultDescriptorResource.texture) {
                this.defaultDescriptorResource.texture.destroy();
            }
            if (this.defaultDescriptorResource.sampler) {
                this.defaultDescriptorResource.sampler.destroy();
            }
        }
    }

    public resize(width: number, height: number) {

    }

    public acquire() {
    }

    get swapChain() {
        return this._swapChain;
    }

    public nativeDevice() {
        return this._device;
    }

    public glslang() {
        return this._glslang;
    }

    public present() {
        const queue = (this._queue as unknown as WebGPUQueue);
        this._numDrawCalls = queue.numDrawCalls;
        this._numInstances = queue.numInstances;
        this._numTris = queue.numTris;
        queue.clear();
    }

    public createCommandBuffer(info: CommandBufferInfo): CommandBuffer {
        const cmdBuff = new WebGPUCommandBuffer(this);
        if (cmdBuff.initialize(info)) {
            return cmdBuff;
        }
        return null!;
    }

    public createBuffer(info: BufferInfo | BufferViewInfo): Buffer {
        const buffer = new WebGPUBuffer(this);
        if (buffer.initialize(info)) {
            return buffer;
        }
        return null!;
    }

    public createTexture(info: TextureInfo | TextureViewInfo): Texture {
        const texture = new WebGPUTexture(this);
        if (texture.initialize(info)) {
            return texture;
        }
        return null!;
    }

    public createSampler(info: SamplerInfo): Sampler {
        const sampler = new WebGPUSampler(this);
        if (sampler.initialize(info)) {
            return sampler;
        }
        return null!;
    }

    public createDescriptorSet(info: DescriptorSetInfo): DescriptorSet {
        const descriptorSet = new WebGPUDescriptorSet(this);
        if (descriptorSet.initialize(info)) {
            return descriptorSet;
        }
        return null!;
    }

    public createShader(info: ShaderInfo): Shader {
        const shader = new WebGPUShader(this);
        if (shader.initialize(info)) {
            return shader;
        }
        return null!;
    }

    public createInputAssembler(info: InputAssemblerInfo): InputAssembler {
        const inputAssembler = new WebGPUInputAssembler(this);
        if (inputAssembler.initialize(info)) {
            return inputAssembler;
        }
        return null!;
    }

    public createRenderPass(info: RenderPassInfo): RenderPass {
        const renderPass = new WebGPURenderPass(this);
        if (renderPass.initialize(info)) {
            return renderPass;
        }
        return null!;
    }

    public createFramebuffer(info: FramebufferInfo): Framebuffer {
        const framebuffer = new WebGPUFramebuffer(this);
        if (framebuffer.initialize(info)) {
            return framebuffer;
        }
        return null!;
    }

    public createDescriptorSetLayout(info: DescriptorSetLayoutInfo): DescriptorSetLayout {
        const descriptorSetLayout = new WebGPUDescriptorSetLayout(this);
        if (descriptorSetLayout.initialize(info)) {
            return descriptorSetLayout;
        }
        return null!;
    }

    public createPipelineLayout(info: PipelineLayoutInfo): PipelineLayout {
        const pipelineLayout = new WebGPUPipelineLayout(this);
        if (pipelineLayout.initialize(info)) {
            return pipelineLayout;
        }
        return null!;
    }

    public createPipelineState(info: PipelineStateInfo): PipelineState {
        const pipelineState = new WebGPUPipelineState(this);
        if (pipelineState.initialize(info)) {
            return pipelineState;
        }
        return null!;
    }

    public createQueue(info: QueueInfo): Queue {
        const queue = new WebGPUQueue(this);
        if (queue.initialize(info)) {
            return queue;
        }
        return null!;
    }

    public copyBuffersToTexture(buffers: ArrayBufferView[], texture: Texture, regions: BufferTextureCopy[]) {
        WebGPUCmdFuncCopyBuffersToTexture(
            this,
            buffers,
            (texture as unknown as WebGPUTexture).gpuTexture,
            regions,
        );
    }

    public copyTexImagesToTexture(
        texImages: TexImageSource[],
        texture: Texture,
        regions: BufferTextureCopy[],
    ) {
        WebGPUCmdFuncCopyTexImagesToTexture(
            this,
            texImages,
            (texture as unknown as WebGPUTexture).gpuTexture,
            regions,
        );
    }

    public copyFramebufferToBuffer(
        srcFramebuffer: Framebuffer,
        dstBuffer: ArrayBuffer,
        regions: BufferTextureCopy[],
    ) { }

    public blitFramebuffer(src: Framebuffer, dst: Framebuffer, srcRect: Rect, dstRect: Rect, filter: Filter) { }
}
