import { IndirectBuffer } from '../buffer';
import { Buffer, BufferSource, BufferInfo, BufferViewInfo } from '../buffer';
import { BufferFlagBit, BufferUsageBit } from '../define';
import {
    WebGPUCmdFuncCreateBuffer,
    WebGPUCmdFuncDestroyBuffer,
    WebGPUCmdFuncResizeBuffer,
    WebGPUCmdFuncUpdateBuffer,
} from './webgpu-commands';
import { WebGPUDevice } from './webgpu-device';
import { IWebGPUGPUBuffer } from './webgpu-gpu-objects';

export class WebGPUBuffer extends Buffer {

    get gpuBuffer (): IWebGPUGPUBuffer {
        return  this._gpuBuffer!;
    }

    private _gpuBuffer: IWebGPUGPUBuffer | null = null;

    public initialize (info: BufferInfo | BufferViewInfo): boolean {

        if ('buffer' in info) { // buffer view

            this._isBufferView = true;

            const buffer = info.buffer as WebGPUBuffer;

            this._usage = buffer.usage;
            this._memUsage = buffer.memUsage;
            this._size = this._stride = info.range;
            this._count = 1;
            this._flags = buffer.flags;

            this._gpuBuffer = {
                usage: this._usage,
                memUsage: this._memUsage,
                size: this._size,
                stride: this._stride,
                buffer: this._bakcupBuffer,
                indirects: buffer.gpuBuffer.indirects,
                glTarget: buffer.gpuBuffer.glTarget,
                glBuffer: buffer.gpuBuffer.glBuffer,
                glOffset: info.offset,
            };

        } else { // native buffer

            this._usage = info.usage;
            this._memUsage = info.memUsage;
            this._size = info.size;
            this._stride = Math.max(info.stride || this._size, 1);
            this._count = this._size / this._stride;
            this._flags = info.flags;

            if (this._usage & BufferUsageBit.INDIRECT) {
                this._indirectBuffer = new IndirectBuffer();
            }

            if (this._flags & BufferFlagBit.BAKUP_BUFFER) {
                this._bakcupBuffer = new Uint8Array(this._size);
                this._device.memoryStatus.bufferSize += this._size;
            }

            this._gpuBuffer = {
                usage: this._usage,
                memUsage: this._memUsage,
                size: this._size,
                stride: this._stride,
                buffer: this._bakcupBuffer,
                indirects: [],
                glTarget: 0,
                glBuffer: null,
                glOffset: 0,
            };

            if (info.usage & BufferUsageBit.INDIRECT) {
                this._gpuBuffer.indirects = this._indirectBuffer!.drawInfos;
            }

            WebGPUCmdFuncCreateBuffer(this._device as WebGPUDevice, this._gpuBuffer);

            this._device.memoryStatus.bufferSize += this._size;
        }

        return true;
    }

    public destroy () {
        if (this._gpuBuffer) {
            if (!this._isBufferView) {
                WebGPUCmdFuncDestroyBuffer(this._device as WebGPUDevice, this._gpuBuffer);
                this._device.memoryStatus.bufferSize -= this._size;
            }
            this._gpuBuffer = null;
        }

        this._bakcupBuffer = null;
    }

    public resize (size: number) {
        if (this._isBufferView) {
            console.warn('cannot resize buffer views!');
            return;
        }

        const oldSize = this._size;
        if (oldSize === size) { return; }

        this._size = size;
        this._count = this._size / this._stride;

        if (this._bakcupBuffer) {
            const oldView = this._bakcupBuffer;
            this._bakcupBuffer = new Uint8Array(this._size);
            this._bakcupBuffer.set(oldView);
            this._device.memoryStatus.bufferSize -= oldSize;
            this._device.memoryStatus.bufferSize += size;
        }

        if (this._gpuBuffer) {
            if (this._bakcupBuffer) {
                this._gpuBuffer.buffer = this._bakcupBuffer;
            }

            this._gpuBuffer.size = size;
            if (size > 0) {
                WebGPUCmdFuncResizeBuffer(this._device as WebGPUDevice, this._gpuBuffer);
                this._device.memoryStatus.bufferSize -= oldSize;
                this._device.memoryStatus.bufferSize += size;
            }
        }
    }

    public update (buffer: BufferSource, offset?: number, size?: number) {
        if (this._isBufferView) {
            console.warn('cannot update through buffer views!');
            return;
        }

        let buffSize: number;
        if (size !== undefined ) {
            buffSize = size;
        } else if (this._usage & BufferUsageBit.INDIRECT) {
            buffSize = 0;
        } else {
            buffSize = (buffer as ArrayBuffer).byteLength;
        }
        if (this._bakcupBuffer && buffer !== this._bakcupBuffer.buffer) {
            const view = new Uint8Array(buffer as ArrayBuffer, 0, size);
            this._bakcupBuffer.set(view, offset);
        }

        WebGPUCmdFuncUpdateBuffer(
            this._device as WebGPUDevice,
            this._gpuBuffer!,
            buffer,
            offset || 0,
            buffSize);
    }
}
