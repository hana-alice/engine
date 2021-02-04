import { color } from '../../math';
import { RenderPass, RenderPassInfo } from '../render-pass';
import { IWebGPUGPURenderPass } from './webgpu-gpu-objects';
import { StoreOp } from '../define';
import {  } from '../../utils/js';
import { WebGPUTexture } from './webgpu-texture';

export class WebGPURenderPass extends RenderPass {

    private _renPassDescriptor: GPURenderPassDescriptor | null = null;

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

        this._gpuRenderPass = {
            colorAttachments: this._colorInfos,
            depthStencilAttachment: this._depthStencilInfo,
        };

        let colorDescriptions: GPURenderPassColorAttachmentDescriptor[] = [];
        for (let attachment of info.colorAttachments) {
            let colorDesc = {} as GPURenderPassColorAttachmentDescriptor;
            colorDesc.loadValue = "load";
            colorDesc.storeOp = attachment.storeOp === StoreOp.STORE ? "store" : "clear";
            
            colorDesc.attachment = {} as GPUTextureView;
            colorDescriptions[colorDescriptions.length] = colorDesc;
        }

        this._renPassDescriptor = {} as GPURenderPassDescriptor;
        this._renPassDescriptor.colorAttachments = colorDescriptions;

        if (info.depthStencilAttachment) { 
            let depthStencilDescriptor = {} as GPURenderPassDepthStencilAttachmentDescriptor;
            depthStencilDescriptor.depthLoadValue = "load";
            depthStencilDescriptor.depthStoreOp = info.depthStencilAttachment?.depthStoreOp === StoreOp.STORE ? "store" : "clear";
            depthStencilDescriptor.stencilLoadValue = "load";
            depthStencilDescriptor.stencilStoreOp = info.depthStencilAttachment?.stencilStoreOp == StoreOp.STORE ? "store" : "clear";
            depthStencilDescriptor.attachment = {} as GPUTextureView;  

            this._renPassDescriptor.depthStencilAttachment = depthStencilDescriptor;
        }

        this._hash = this.computeHash();

        return true;
    }

    public destroy () {
        this._gpuRenderPass = null;
    }
}
