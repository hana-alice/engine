import { murmurhash2_32_gc } from '../../utils/murmurhash2_gc';
import { Fence, FenceInfo } from '../fence';
import { WebGPUDevice } from './webgpu-device';

export class WebGPUFence extends Fence {
    private _sync: WebGLSync | null = null;
    private _fence: GPUFence | undefined = undefined;
    private _fenceDescriptor: GPUFenceDescriptor | undefined = undefined;
    private _hash = 0;

    public initialize (info: FenceInfo): boolean {
        const label = 'now fence for pipeline only, make this sentence a type-unique label if numlti-fence needed.';
        this._hash = murmurhash2_32_gc(label, 666);
        this._fenceDescriptor = {
            initialValue: this._hash,
            label,
            signalQueue: undefined,
        };
        return true;
    }

    public aquire () {
        this._fence = (this._device as WebGPUDevice).nativeDevice()?.queue.createFence(this._fenceDescriptor);
    }

    public release () {
        (this._device as WebGPUDevice).nativeDevice()?.queue.signal(this._fence!, 0);
    }

    public destroy () {
    }

    public insert () {

    }
}
