import { color } from '../../math';
import { RenderPass, RenderPassInfo } from '../render-pass';
import { IWebGPUGPURenderPass } from './webgpu-gpu-objects';
import { StoreOp } from '../define';
import {  } from '../../utils/js';

export class WebGPURenderPass extends RenderPass {
    public get gpuRenderPass (): IWebGPUGPURenderPass {
        return  this._gpuRenderPass!;
    }

    private _gpuRenderPass: IWebGPUGPURenderPass | null = null;

    public initialize (info: RenderPassInfo): boolean {
        this._colorInfos = info.colorAttachments;
        this._depthStencilInfo = info.depthStencilAttachment;
        if (info.subPasses) {
            this._subPasses = info.subPasses;
        }

        const renderPassDesc = {} as GPURenderPassDescriptor;

        const colorDescriptions: GPURenderPassColorAttachmentDescriptor[] = [];
        for (const attachment of info.colorAttachments) {
            colorDescriptions[colorDescriptions.length] = {
                attachment: {} as GPUTextureView,
                loadValue: 'load',
                storeOp: attachment.storeOp === StoreOp.STORE ? 'store' : 'clear',
            };
        }
        renderPassDesc.colorAttachments = colorDescriptions;

        if (info.depthStencilAttachment) {
            const depthStencilDescriptor = {} as GPURenderPassDepthStencilAttachmentDescriptor;
            depthStencilDescriptor.depthLoadValue = 'load';
            depthStencilDescriptor.depthStoreOp = info.depthStencilAttachment?.depthStoreOp === StoreOp.STORE ? 'store' : 'clear';
            depthStencilDescriptor.stencilLoadValue = 'load';
            depthStencilDescriptor.stencilStoreOp = info.depthStencilAttachment?.stencilStoreOp === StoreOp.STORE ? 'store' : 'clear';
            depthStencilDescriptor.attachment = {} as GPUTextureView;

            renderPassDesc.depthStencilAttachment = {
                attachment: {} as GPUTextureView,
                depthLoadValue: 'load',
                depthStoreOp: info.depthStencilAttachment?.depthStoreOp === StoreOp.STORE ? 'store' : 'clear',
                stencilLoadValue: 'load',
                stencilStoreOp: info.depthStencilAttachment?.stencilStoreOp === StoreOp.STORE ? 'store' : 'clear',
            };
        }

        this._gpuRenderPass = {
            colorAttachments: this._colorInfos,
            depthStencilAttachment: this._depthStencilInfo,
            nativeRenderPass: renderPassDesc,
        };

        this._hash = this.computeHash();

        return true;
    }

    public destroy () {
        this._gpuRenderPass = null;
    }
}
