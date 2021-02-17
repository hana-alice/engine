import { Framebuffer, FramebufferInfo } from '../framebuffer';
import { WebGPUCmdFuncCreateFramebuffer, WebGPUCmdFuncDestroyFramebuffer } from './webgpu-commands';
import { WebGPUDevice } from './webgpu-device';
import { IWebGPUGPUFramebuffer, IWebGPUGPUTexture } from './webgpu-gpu-objects';
import { WebGPURenderPass } from './webgpu-render-pass';
import { WebGPUTexture } from './webgpu-texture';

export class WebGPUFramebuffer extends Framebuffer {
    get gpuFramebuffer (): IWebGPUGPUFramebuffer {
        return  this._gpuFramebuffer!;
    }

    private _gpuFramebuffer: IWebGPUGPUFramebuffer | null = null;
    public initialize (info: FramebufferInfo): boolean {
        this._renderPass = info.renderPass;
        this._colorTextures = info.colorTextures || [];
        this._depthStencilTexture = info.depthStencilTexture || null;

        if (info.depStencilMipmapLevel !== 0) {
            console.warn('The mipmap level of th texture image to be attached of depth stencil attachment should be 0. Convert to 0.');
        }
        for (let i = 0; i < info.colorMipmapLevels.length; ++i) {
            if (info.colorMipmapLevels[i] !== 0) {
                console.warn(`The mipmap level of th texture image to be attached of color attachment ${i} should be 0. Convert to 0.`);
            }
        }

        const gpuColorTextures: IWebGPUGPUTexture[] = [];
        let isOffscreen = false;
        // onscreen
        if (info.colorTextures.every((tex) => tex === null)) {
            isOffscreen = false;
        } else { // offscreen
            isOffscreen = true;
            for (let i = 0; i < info.colorTextures.length; i++) {
                const colorTexture = info.colorTextures[i];
                if (colorTexture) {
                    gpuColorTextures.push((colorTexture as WebGPUTexture).gpuTexture);
                }
            }
        }

        let gpuDepthStencilTexture: IWebGPUGPUTexture | null = null;
        if (info.depthStencilTexture) {
            gpuDepthStencilTexture = (info.depthStencilTexture as WebGPUTexture).gpuTexture;
        }

        this._gpuFramebuffer = {
            gpuRenderPass: (info.renderPass as WebGPURenderPass).gpuRenderPass,
            gpuColorTextures,
            gpuDepthStencilTexture,
            glFramebuffer: null,
            isOffscreen,
        };

        // WebGPUCmdFuncCreateFramebuffer(this._device as WebGPUDevice, this._gpuFramebuffer);

        return true;
    }

    public destroy () {
        if (this._gpuFramebuffer) {
            // WebGPUCmdFuncDestroyFramebuffer(this._device as WebGPUDevice, this._gpuFramebuffer);
            this._gpuFramebuffer = null;
        }
    }
}
