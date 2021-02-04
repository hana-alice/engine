import { Fence, FenceInfo } from '../fence';
import { WebGPUDevice } from './webgpu-device';

export class WebGPUFence extends Fence {

    private _sync: WebGLSync | null = null;

    public initialize (info: FenceInfo): boolean {
        return true;
    }

    public destroy () {
    }

    public wait () {
        if (this._sync) {
            const gl = (this._device as WebGPUDevice).gl;
            gl.clientWaitSync(this._sync, 0, gl.TIMEOUT_IGNORED);
        }
    }

    public reset () {
        if (this._sync) {
            const gl = (this._device as WebGPUDevice).gl;
            gl.deleteSync(this._sync);
            this._sync = null;
        }
    }

    public insert () {
        const gl = (this._device as WebGPUDevice).gl;
        if (this._sync) { gl.deleteSync(this._sync); }
        this._sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
}
