import { CommandBuffer } from '../command-buffer';
import { Queue, QueueInfo } from '../queue';
import { WebGPUCommandBuffer } from './webgpu-command-buffer';
import { Fence } from '../fence';
import { WebGPUFence } from './webgpu-fence';
import { WebGPUCmdFuncExecuteCmds } from './webgpu-commands';

export class WebGPUQueue extends Queue {
    public numDrawCalls = 0;
    public numInstances = 0;
    public numTris = 0;

    private _nativeQueue: GPUQueue | null = null;

    public initialize (info: QueueInfo): boolean {
        this._type = info.type;

        return true;
    }

    public destroy () {
    }

    public submit (cmdBuffs: CommandBuffer[], fence?: Fence) {
        // TODO: Async
        if (!this._isAsync) {
            for (let i = 0; i < cmdBuffs.length; i++) {
                const cmdBuff = cmdBuffs[i] as WebGPUCommandBuffer;
                // WebGPUCmdFuncExecuteCmds(this._device as WebGPUDevice, cmdBuff.cmdPackage); // opted out
                this.numDrawCalls += cmdBuff.numDrawCalls;
                this.numInstances += cmdBuff.numInstances;
                this.numTris += cmdBuff.numTris;
            }
        }
        if (fence) {
            (fence as WebGPUFence).insert();
        }
    }

    public clear () {
        this.numDrawCalls = 0;
        this.numInstances = 0;
        this.numTris = 0;
    }
}
