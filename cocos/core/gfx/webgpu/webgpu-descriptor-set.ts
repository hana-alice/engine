import { DescriptorSet, DescriptorSetInfo, DESCRIPTOR_BUFFER_TYPE, DESCRIPTOR_SAMPLER_TYPE } from '../descriptor-set';
import { WebGPUBuffer } from './webgpu-buffer';
import { IWebGPUGPUDescriptorSet, IWebGPUGPUDescriptor } from './webgpu-gpu-objects';
import { WebGPUSampler } from './webgpu-sampler';
import { WebGPUTexture } from './webgpu-texture';
import { WebGPUDescriptorSetLayout } from './webgpu-descriptor-set-layout';
import { WebGPUDevice } from './webgpu-device';
import { DescriptorType } from '../define';

export class WebGPUDescriptorSet extends DescriptorSet {
    get gpuDescriptorSet (): IWebGPUGPUDescriptorSet {
        return this._gpuDescriptorSet as IWebGPUGPUDescriptorSet;
    }

    private _gpuDescriptorSet: IWebGPUGPUDescriptorSet | null = null;
    private _bindGroupEntries: GPUBindGroupEntry[] = [];

    public initialize (info: DescriptorSetInfo): boolean {
        this._layout = info.layout;
        const { bindings, descriptorIndices, descriptorCount } = (info.layout as WebGPUDescriptorSetLayout).gpuDescriptorSetLayout;

        this._buffers = Array(descriptorCount).fill(null);
        this._textures = Array(descriptorCount).fill(null);
        this._samplers = Array(descriptorCount).fill(null);

        const gpuDescriptors: IWebGPUGPUDescriptor[] = [];
        const bindGroup = {} as GPUBindGroup;
        this._gpuDescriptorSet = { gpuDescriptors, descriptorIndices, bindGroup };

        for (let i = 0; i < bindings.length; ++i) {
            const binding = bindings[i];
            for (let j = 0; j < binding.count; j++) {
                if (binding.descriptorType !== DescriptorType.UNKNOWN) {
                    gpuDescriptors.push({
                        type: binding.descriptorType,
                        gpuBuffer: null,
                        gpuTexture: null,
                        gpuSampler: null,
                    });
                }
            }
        }

        return true;
    }

    public destroy () {
        this._layout = null;
        this._gpuDescriptorSet = null;
    }

    public update () {
        if (this._isDirty && this._gpuDescriptorSet) {
            const descriptors = this._gpuDescriptorSet.gpuDescriptors;
            if (descriptors) {
                let count = 0;
                for (let i = 0; i < descriptors.length; ++i) {
                    const binding = (this._layout as WebGPUDescriptorSetLayout).gpuDescriptorSetLayout.bindings[i];
                    if (descriptors[i].type & DESCRIPTOR_BUFFER_TYPE) {
                        if (this._buffers[i]) {
                            descriptors[i].gpuBuffer = (this._buffers[i] as WebGPUBuffer).gpuBuffer;
                            const nativeBuffer = descriptors[i].gpuBuffer?.glBuffer;

                            const bindGrpEntry: GPUBindGroupEntry = {
                                binding: binding.binding,
                                resource: {
                                    buffer: nativeBuffer!,
                                    offset: descriptors[i].gpuBuffer?.glOffset,
                                    size: descriptors[i].gpuBuffer?.size,
                                },
                            };

                            this._bindGroupEntries[count++] = bindGrpEntry;
                        }
                    } else if (descriptors[i].type & DESCRIPTOR_SAMPLER_TYPE) {
                        if (this._textures[i]) {
                            descriptors[i].gpuTexture = (this._textures[i] as WebGPUTexture).gpuTexture;

                            const bindGrpEntry: GPUBindGroupEntry = {
                                binding: binding.binding,
                                resource: descriptors[i].gpuTexture?.glTexture?.createView() as GPUTextureView,
                            };

                            this._bindGroupEntries[count++] = bindGrpEntry;
                        }
                        if (this._samplers[i]) {
                            descriptors[i].gpuSampler = (this._samplers[i] as WebGPUSampler).gpuSampler;
                            const bindGrpEntry: GPUBindGroupEntry = {
                                binding: binding.binding,
                                resource: descriptors[i].gpuSampler?.glSampler as GPUSampler,
                            };
                            this._bindGroupEntries[count++] = bindGrpEntry;
                        }
                    }
                }
                this._isDirty = false;
                const nativeDevice = (this._device as WebGPUDevice).nativeDevice();
                const bindGroup = nativeDevice?.createBindGroup({
                    layout: (this._layout as WebGPUDescriptorSetLayout).gpuDescriptorSetLayout.bindGroupLayout,
                    entries: this._bindGroupEntries,
                });

                this._gpuDescriptorSet.bindGroup = bindGroup!;
            }
        }
    }
}
