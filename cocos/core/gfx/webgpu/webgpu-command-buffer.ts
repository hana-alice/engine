import { DescriptorSet } from '../descriptor-set';
import { Buffer, BufferSource, DrawInfo } from '../buffer';
import { CommandBuffer, CommandBufferInfo } from '../command-buffer';
import {
    BufferUsageBit,
    CommandBufferType,
    StencilFace,
    StoreOp,
} from '../define';
import { BufferTextureCopy, Color, Rect, Viewport } from '../define-class';
import { Framebuffer } from '../framebuffer';
import { InputAssembler } from '../input-assembler';
import { PipelineState } from '../pipeline-state';
import { Texture } from '../texture';
import { WebGPUDescriptorSet } from './webgpu-descriptor-set';
import { WebGPUBuffer } from './webgpu-buffer';
import { WebGPUCommandAllocator } from './webgpu-command-allocator';
import {
    WebGPUCmd,
    WebGPUCmdBeginRenderPass,
    WebGPUCmdBindStates,
    WebGPUCmdCopyBufferToTexture,
    WebGPUCmdDraw,
    WebGPUCmdPackage,
    WebGPUCmdUpdateBuffer,
    WebGPUCmdFuncBeginRenderPass,
} from './webgpu-commands';
import { WebGPUDevice } from './webgpu-device';
import { WebGPUFramebuffer } from './webgpu-framebuffer';
import { IWebGPUGPUInputAssembler,
    IWebGPUGPUDescriptorSet,
    IWebGPUGPUPipelineState,
    IWebGPUGPUPipelineLayout,
    IWebGPUGPUBuffer,
} from './webgpu-gpu-objects';
import { WebGPUInputAssembler } from './webgpu-input-assembler';
import { WebGPUPipelineState } from './webgpu-pipeline-state';
import { WebGPUTexture } from './webgpu-texture';
import { RenderPass } from '../render-pass';
import { WebGPURenderPass } from './webgpu-render-pass';
import { WebGPUFence } from './webgpu-fence';
import { view } from '../../platform';
import { INT_MAX } from '../../math/bits';
import { size } from '../../math';
import { barFilled } from '../../../2d/assembler';

export interface IWebGPUDepthBias {
    constantFactor: number;
    clamp: number;
    slopeFactor: number;
}

export interface IWebGPUDepthBounds {
    minBounds: number;
    maxBounds: number;
}

export interface IWebGPUStencilWriteMask {
    face: StencilFace;
    writeMask: number;
}

export interface IWebGPUStencilCompareMask {
    face: StencilFace;
    reference: number;
    compareMask: number;
}

type CommandEncoder = {commandEncoder: GPUCommandEncoder, renderPassEncoder: GPURenderPassEncoder};
export class WebGPUCommandBuffer extends CommandBuffer {
    public cmdPackage: WebGPUCmdPackage = new WebGPUCmdPackage();
    protected _webGLAllocator: WebGPUCommandAllocator | null = null;
    protected _isInRenderPass = false;
    protected _curGPUPipelineState: IWebGPUGPUPipelineState | null = null;
    protected _curGPUDescriptorSets: IWebGPUGPUDescriptorSet[] = [];
    protected _curGPUInputAssembler: IWebGPUGPUInputAssembler | null = null;
    protected _curDynamicOffsets: number[][] = [];
    protected _curViewport: Viewport | null = null;
    protected _curScissor: Rect | null = null;
    protected _curLineWidth: number | null = null;
    protected _curDepthBias: IWebGPUDepthBias | null = null;
    protected _curBlendConstants: number[] = [0.0, 0.0, 0.0, 0.0];
    protected _curDepthBounds: IWebGPUDepthBounds | null = null;
    protected _curStencilWriteMask: IWebGPUStencilWriteMask | null = null;
    protected _curStencilCompareMask: IWebGPUStencilCompareMask | null = null;
    protected _isStateInvalied = false;

    private _nativeCommandBuffer: GPUCommandBuffer | null = null;
    private _fence: WebGPUFence | null = null;
    private _encoder: CommandEncoder | undefined = undefined;
    private _descSetDirtyIndex: number = INT_MAX;

    public initialize (info: CommandBufferInfo): boolean {
        this._type = info.type;
        this._queue = info.queue;

        this._webGLAllocator = (this._device as WebGPUDevice).cmdAllocator;
        this._encoder = {} as CommandEncoder;

        const setCount = (this._device as WebGPUDevice).bindingMappingInfo.bufferOffsets.length;
        for (let i = 0; i < setCount; i++) {
            this._curGPUDescriptorSets.push(null!);
            this._curDynamicOffsets.push([]);
        }

        return true;
    }

    public destroy () {
        if (this._webGLAllocator) {
            this._webGLAllocator.clearCmds(this.cmdPackage);
            this._webGLAllocator = null;
        }
    }

    public begin (renderPass?: RenderPass, subpass = 0, frameBuffer?: Framebuffer) {
        this._webGLAllocator!.clearCmds(this.cmdPackage);
        this._curGPUPipelineState = null;
        this._curGPUInputAssembler = null;
        this._curGPUDescriptorSets.length = 0;
        for (let i = 0; i < this._curDynamicOffsets.length; i++) {
            this._curDynamicOffsets[i].length = 0;
        }
        this._curViewport = null;
        this._curScissor = null;
        this._curLineWidth = null;
        this._curDepthBias = null;
        // this._curBlendConstants.length = 0;
        this._curDepthBounds = null;
        this._curStencilWriteMask = null;
        this._curStencilCompareMask = null;
        this._numDrawCalls = 0;
        this._numInstances = 0;
        this._numTris = 0;
    }

    public end () {
        if (this._isStateInvalied) {
            this.bindStates();
        }

        this._isInRenderPass = false;
    }

    public beginRenderPass (
        renderPass: RenderPass,
        framebuffer: Framebuffer,
        renderArea: Rect,
        clearColors: Color[],
        clearDepth: number,
        clearStencil: number,
    ) {
        const gpuDevice = this._device as WebGPUDevice;
        const nativeRenPassDesc = (renderPass as WebGPURenderPass).gpuRenderPass.nativeRenderPass!;
        const gpuFramebuffer = (framebuffer as WebGPUFramebuffer).gpuFramebuffer;
        const gpuRenderPass = renderPass as WebGPURenderPass;

        for (let i = 0; i < clearColors.length; i++) {
            const colorTex = gpuFramebuffer.isOffscreen ? gpuFramebuffer.gpuColorTextures[i].glTexture?.createView()
                : gpuDevice.defaultColorTex?.createView();
            nativeRenPassDesc.colorAttachments[i] = {
                attachment: colorTex,
                loadValue: [clearColors[i].x, clearColors[i].y, clearColors[i].z, clearColors[i].w], // RGBA
                storeOp: gpuRenderPass?.colorAttachments[i].storeOp === StoreOp.STORE ? 'store' : 'clear',
            };
        }

        if (gpuRenderPass?.depthStencilAttachment) {
            const tex = gpuFramebuffer.gpuDepthStencilTexture?.glTexture;
            const depthTex = tex ? tex.createView() : gpuDevice.defaultDepthStencilTex?.createView();
            nativeRenPassDesc.depthStencilAttachment = {
                attachment: depthTex!,
                depthLoadValue: clearDepth,
                depthStoreOp: gpuRenderPass.depthStencilAttachment.depthStoreOp === StoreOp.STORE ? 'store' : 'clear',
                // depthReadOnly:
                stencilLoadValue: clearStencil,
                stencilStoreOp: gpuRenderPass.depthStencilAttachment.stencilStoreOp === StoreOp.STORE ? 'store' : 'clear',
                // stencilReadOnly:
            };
        }

        const cmdEncoder = gpuDevice.nativeDevice()?.createCommandEncoder();
        const renpassEncoder = cmdEncoder?.beginRenderPass(nativeRenPassDesc);
        this._encoder = {
            commandEncoder: cmdEncoder as GPUCommandEncoder,
            renderPassEncoder: renpassEncoder!,
        };

        this._encoder?.renderPassEncoder.setViewport(renderArea.x, renderArea.y, renderArea.width, renderArea.height, 0.0, 1.0);
        this._encoder?.renderPassEncoder.setScissorRect(renderArea.x, renderArea.y, renderArea.width, renderArea.height);
        this._isInRenderPass = true;
    }

    public endRenderPass () {
        this._encoder?.renderPassEncoder.endPass();
        const nativeDevice = (this._device as WebGPUDevice).nativeDevice();
        nativeDevice?.defaultQueue.submit([this._encoder?.commandEncoder.finish() as GPUCommandBuffer]);
        this._isInRenderPass = false;
    }

    public bindPipelineState (pipelineState: PipelineState) {
        const gpuPipelineState = (pipelineState as WebGPUPipelineState).gpuPipelineState;
        if (gpuPipelineState !== this._curGPUPipelineState) {
            this._curGPUPipelineState = gpuPipelineState;
            this._isStateInvalied = true;
        }
    }

    public bindDescriptorSet (set: number, descriptorSet: DescriptorSet, dynamicOffsets?: number[]) {
        const gpuDescriptorSets = (descriptorSet as WebGPUDescriptorSet).gpuDescriptorSet;
        if (gpuDescriptorSets !== this._curGPUDescriptorSets[set]) {
            this._curGPUDescriptorSets[set] = gpuDescriptorSets;
            this._isStateInvalied = true;
        }
        if (dynamicOffsets) {
            const offsets = this._curDynamicOffsets[set];
            for (let i = 0; i < dynamicOffsets.length; i++) offsets[i] = dynamicOffsets[i];
            offsets.length = dynamicOffsets.length;
            this._isStateInvalied = true;
        }
    }

    public bindInputAssembler (inputAssembler: InputAssembler) {
        const gpuInputAssembler = (inputAssembler as WebGPUInputAssembler).gpuInputAssembler;
        this._curGPUInputAssembler = gpuInputAssembler;
        this._isStateInvalied = true;
    }

    public setViewport (viewport: Viewport) {
        this._curViewport = new Viewport(viewport.left, viewport.top, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth);
        // FIXME: what viewport.top exactly means?
        this._encoder?.renderPassEncoder.setViewport(viewport.left, viewport.top,
            viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth);
    }

    public setScissor (scissor: Rect) {
        this._encoder?.renderPassEncoder.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height);
    }

    public setLineWidth (lineWidth: number) {
        console.error('line width not supproted by webGPU');
    }

    public setDepthBias (depthBiasConstantFactor: number, depthBiasClamp: number, depthBiasSlopeFactor: number) {
        if (!this._curDepthBias) {
            this._curDepthBias = {
                constantFactor: depthBiasConstantFactor,
                clamp: depthBiasClamp,
                slopeFactor: depthBiasSlopeFactor,
            };
            this._isStateInvalied = true;
        } else if (this._curDepthBias.constantFactor !== depthBiasConstantFactor
                || this._curDepthBias.clamp !== depthBiasClamp
                || this._curDepthBias.slopeFactor !== depthBiasSlopeFactor) {
            this._curDepthBias.constantFactor = depthBiasConstantFactor;
            this._curDepthBias.clamp = depthBiasClamp;
            this._curDepthBias.slopeFactor = depthBiasSlopeFactor;
            this._isStateInvalied = true;
        }
    }

    public setBlendConstants (blendConstants: number[]) {
        if (blendConstants.length === 4 && (
            this._curBlendConstants[0] !== blendConstants[0]
            || this._curBlendConstants[1] !== blendConstants[1]
            || this._curBlendConstants[2] !== blendConstants[2]
            || this._curBlendConstants[3] !== blendConstants[3])) {
            this._curBlendConstants.length = 0;
            Array.prototype.push.apply(this._curBlendConstants, blendConstants);
            this._isStateInvalied = true;
        }
    }

    public setDepthBound (minDepthBounds: number, maxDepthBounds: number) {
        if (!this._curDepthBounds) {
            this._curDepthBounds = {
                minBounds: minDepthBounds,
                maxBounds: maxDepthBounds,
            };
            this._isStateInvalied = true;
        } else if (this._curDepthBounds.minBounds !== minDepthBounds
                || this._curDepthBounds.maxBounds !== maxDepthBounds) {
            this._curDepthBounds = {
                minBounds: minDepthBounds,
                maxBounds: maxDepthBounds,
            };
            this._isStateInvalied = true;
        }
    }

    public setStencilWriteMask (face: StencilFace, writeMask: number) {
        if (!this._curStencilWriteMask) {
            this._curStencilWriteMask = {
                face,
                writeMask,
            };
            this._isStateInvalied = true;
        } else if (this._curStencilWriteMask.face !== face
                || this._curStencilWriteMask.writeMask !== writeMask) {
            this._curStencilWriteMask.face = face;
            this._curStencilWriteMask.writeMask = writeMask;
            this._isStateInvalied = true;
        }
    }

    public setStencilCompareMask (face: StencilFace, reference: number, compareMask: number) {
        if (!this._curStencilCompareMask) {
            this._curStencilCompareMask = {
                face,
                reference,
                compareMask,
            };
            this._isStateInvalied = true;
        } else if (this._curStencilCompareMask.face !== face
                || this._curStencilCompareMask.reference !== reference
                || this._curStencilCompareMask.compareMask !== compareMask) {
            this._curStencilCompareMask.face = face;
            this._curStencilCompareMask.reference = reference;
            this._curStencilCompareMask.compareMask = compareMask;
            this._isStateInvalied = true;
        }
    }

    public draw (inputAssembler: InputAssembler) {
        if (this._type === CommandBufferType.PRIMARY && this._isInRenderPass
            || this._type === CommandBufferType.SECONDARY) {
            if (this._isStateInvalied) {
                this.bindStates();
            }

            const passEncoder = this._encoder?.renderPassEncoder;
            const ia = inputAssembler as WebGPUInputAssembler;
            const iaData = ia.gpuInputAssembler;
            const nativeDevice = this._device as WebGPUDevice;

            if (ia.indirectBuffer) {
                const indirectBuffer = iaData.gpuIndirectBuffer!;
                if (nativeDevice.multiDrawIndirectSupport) {
                    // not support yet
                } else {
                    const drawInfoCount = iaData.gpuIndirectBuffer?.indirects.length as number;
                    if (indirectBuffer.drawIndirectByIndex) {
                        for (let i = 0; i < drawInfoCount; i++) {
                            passEncoder?.drawIndexedIndirect(indirectBuffer.glBuffer!, indirectBuffer.glOffset + i * Object.keys(DrawInfo).length);
                        }
                    } else {
                        // FIXME: draw IndexedIndirect and Indirect by different buffer
                        for (let i = 0; i < drawInfoCount; i++) {
                            passEncoder?.drawIndirect(indirectBuffer.glBuffer!, indirectBuffer.glOffset + i * Object.keys(DrawInfo).length);
                        }
                    }
                }
            } else {
                const instanceCount = inputAssembler.instanceCount > 0 ? inputAssembler.instanceCount : 1;
                const drawByIndex = inputAssembler.indirectBuffer && (ia.indexCount > 0);

                if (drawByIndex) {
                    passEncoder?.drawIndexed(ia.indexCount, ia.instanceCount, ia.firstIndex, ia.firstIndex, ia.firstInstance);
                } else {
                    passEncoder?.draw(ia.vertexCount, ia.instanceCount, ia.firstVertex, ia.firstInstance);
                }
            }

            passEncoder?.endPass();
            const commandEncoder = this._encoder?.commandEncoder as GPUCommandEncoder;
            nativeDevice.nativeDevice()?.defaultQueue.submit([commandEncoder.finish()]);

            ++this._numDrawCalls;
            this._numInstances += inputAssembler.instanceCount;
            const indexCount = inputAssembler.indexCount || inputAssembler.vertexCount;
            if (this._curGPUPipelineState) {
                const glPrimitive = this._curGPUPipelineState.glPrimitive;
                switch (glPrimitive) {
                case 'triangle-strip':
                    this._numTris += (indexCount - 2) * Math.max(inputAssembler.instanceCount, 1);
                    break;
                case 'triangle-list': { // WebGLRenderingContext.TRIANGLES
                    this._numTris += indexCount / 3 * Math.max(inputAssembler.instanceCount, 1);
                    break;
                }
                default:
                    break;
                }
            }
        } else {
            console.error('Command \'draw\' must be recorded inside a render pass.');
        }
    }

    public updateBuffer (buffer: Buffer, data: BufferSource, offset?: number, size?: number) {
        if (this._type === CommandBufferType.PRIMARY && !this._isInRenderPass
            || this._type === CommandBufferType.SECONDARY) {
            const gpuBuffer = (buffer as WebGPUBuffer).gpuBuffer;
            if (gpuBuffer) {
                const cmd = this._webGLAllocator!.updateBufferCmdPool.alloc(WebGPUCmdUpdateBuffer);
                let buffSize = 0;
                let buff: BufferSource | null = null;

                // TODO: Have to copy to staging buffer first to make this work for the execution is deferred.
                // But since we are using specialized primary command buffers in WebGL backends, we leave it as is for now
                if (buffer.usage & BufferUsageBit.INDIRECT) {
                    buff = data;
                } else {
                    if (size !== undefined) {
                        buffSize = size;
                    } else {
                        buffSize = (data as ArrayBuffer).byteLength;
                    }
                    buff = data;
                }

                cmd.gpuBuffer = gpuBuffer;
                cmd.buffer = buff;
                cmd.offset = (offset !== undefined ? offset : 0);
                cmd.size = buffSize;
                this.cmdPackage.updateBufferCmds.push(cmd);

                this.cmdPackage.cmds.push(WebGPUCmd.UPDATE_BUFFER);
            }
        } else {
            console.error('Command \'updateBuffer\' must be recorded outside a render pass.');
        }
    }

    public copyBuffersToTexture (buffers: ArrayBufferView[], texture: Texture, regions: BufferTextureCopy[]) {
        if (this._type === CommandBufferType.PRIMARY && !this._isInRenderPass
            || this._type === CommandBufferType.SECONDARY) {
            const gpuTexture = (texture as WebGPUTexture).gpuTexture;
            if (gpuTexture) {
                const cmd = this._webGLAllocator!.copyBufferToTextureCmdPool.alloc(WebGPUCmdCopyBufferToTexture);
                cmd.gpuTexture = gpuTexture;
                cmd.regions = regions;
                // TODO: Have to copy to staging buffer first to make this work for the execution is deferred.
                // But since we are using specialized primary command buffers in WebGL backends, we leave it as is for now
                cmd.buffers = buffers;

                this.cmdPackage.copyBufferToTextureCmds.push(cmd);
                this.cmdPackage.cmds.push(WebGPUCmd.COPY_BUFFER_TO_TEXTURE);
            }
        } else {
            console.error('Command \'copyBufferToTexture\' must be recorded outside a render pass.');
        }
    }

    public execute (cmdBuffs: CommandBuffer[], count: number) {
        for (let i = 0; i < count; ++i) {
            const WebGPUCmdBuff = cmdBuffs[i] as WebGPUCommandBuffer;

            for (let c = 0; c < WebGPUCmdBuff.cmdPackage.beginRenderPassCmds.length; ++c) {
                const cmd = WebGPUCmdBuff.cmdPackage.beginRenderPassCmds.array[c];
                ++cmd.refCount;
                this.cmdPackage.beginRenderPassCmds.push(cmd);
            }

            for (let c = 0; c < WebGPUCmdBuff.cmdPackage.bindStatesCmds.length; ++c) {
                const cmd = WebGPUCmdBuff.cmdPackage.bindStatesCmds.array[c];
                ++cmd.refCount;
                this.cmdPackage.bindStatesCmds.push(cmd);
            }

            for (let c = 0; c < WebGPUCmdBuff.cmdPackage.drawCmds.length; ++c) {
                const cmd = WebGPUCmdBuff.cmdPackage.drawCmds.array[c];
                ++cmd.refCount;
                this.cmdPackage.drawCmds.push(cmd);
            }

            for (let c = 0; c < WebGPUCmdBuff.cmdPackage.updateBufferCmds.length; ++c) {
                const cmd = WebGPUCmdBuff.cmdPackage.updateBufferCmds.array[c];
                ++cmd.refCount;
                this.cmdPackage.updateBufferCmds.push(cmd);
            }

            for (let c = 0; c < WebGPUCmdBuff.cmdPackage.copyBufferToTextureCmds.length; ++c) {
                const cmd = WebGPUCmdBuff.cmdPackage.copyBufferToTextureCmds.array[c];
                ++cmd.refCount;
                this.cmdPackage.copyBufferToTextureCmds.push(cmd);
            }

            this.cmdPackage.cmds.concat(WebGPUCmdBuff.cmdPackage.cmds.array);

            this._numDrawCalls += WebGPUCmdBuff._numDrawCalls;
            this._numInstances += WebGPUCmdBuff._numInstances;
            this._numTris += WebGPUCmdBuff._numTris;
        }
    }

    public get webGLDevice (): WebGPUDevice {
        return this._device as WebGPUDevice;
    }

    protected bindStates () {
        const { dynamicOffsetIndices } = this._curGPUPipelineState?.gpuPipelineLayout as IWebGPUGPUPipelineLayout;
        for (let i = 0; i < this._curGPUDescriptorSets.length; i++) {
            // FIXME: this is a special sentence that 2 in 3 parameters I'm not certain.
            this._encoder?.renderPassEncoder.setBindGroup(i, this._curGPUDescriptorSets[i].bindGroup, this._curDynamicOffsets[i]);
        }

        const ia = this._curGPUInputAssembler!;

        for (let i = 0; i < ia.gpuVertexBuffers.length; i++) {
            this._encoder?.renderPassEncoder.setVertexBuffer(i, ia.gpuVertexBuffers[i].glBuffer!, ia.gpuVertexBuffers[i].glOffset);
        }

        this._encoder?.renderPassEncoder.setBlendColor([this._curBlendConstants[0],
            this._curBlendConstants[1],
            this._curBlendConstants[2],
            this._curBlendConstants[3]]);

        this._encoder?.renderPassEncoder.setIndexBuffer(ia.gpuIndexBuffer?.glBuffer as GPUBuffer,
            ia.glIndexType, ia.gpuIndexBuffer?.glOffset, ia.gpuIndexBuffer?.size);

        this._isStateInvalied = false;
    }
}
