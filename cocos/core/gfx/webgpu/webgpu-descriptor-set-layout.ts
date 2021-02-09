import { DescriptorSetLayout, DescriptorSetLayoutInfo, DESCRIPTOR_DYNAMIC_TYPE } from '../descriptor-set-layout';
import { IWebGPUGPUDescriptorSetLayout } from './webgpu-gpu-objects';
import { WebGPUDevice } from './webgpu-device';
import { GLStageToWebGPUStage, GLDescTypeToWebGPUDescType } from './webgpu-commands';

export class WebGPUDescriptorSetLayout extends DescriptorSetLayout {
    get gpuDescriptorSetLayout () { return this._gpuDescriptorSetLayout!; }

    private _gpuDescriptorSetLayout: IWebGPUGPUDescriptorSetLayout | null = null;

    public initialize (info: DescriptorSetLayoutInfo) {
        Array.prototype.push.apply(this._bindings, info.bindings);

        const nativeDevice = (this._device as WebGPUDevice).nativeDevice();

        let descriptorCount = 0; let maxBinding = -1;
        const flattenedIndices: number[] = [];
        for (let i = 0; i < this._bindings.length; i++) {
            const binding = this._bindings[i];
            flattenedIndices.push(descriptorCount);
            descriptorCount += binding.count;
            if (binding.binding > maxBinding) maxBinding = binding.binding;
        }

        this._bindingIndices = Array(maxBinding + 1).fill(-1);
        const descriptorIndices = this._descriptorIndices = Array(maxBinding + 1).fill(-1);
        const bindGrpLayoutEntries: GPUBindGroupLayoutEntry[] = [];
        for (let i = 0; i < this._bindings.length; i++) {
            const binding = this._bindings[i];
            this._bindingIndices[binding.binding] = i;
            descriptorIndices[binding.binding] = flattenedIndices[i];

            const grpLayoutEntry: GPUBindGroupLayoutEntry = {
                binding: binding.binding,
                visibility: GLStageToWebGPUStage(binding.stageFlags),
                type: GLDescTypeToWebGPUDescType(binding.descriptorType)!,
            };
            bindGrpLayoutEntries.push(grpLayoutEntry);
        }

        const bindGrpLayout = nativeDevice?.createBindGroupLayout({ entries: bindGrpLayoutEntries });

        const dynamicBindings: number[] = [];
        for (let i = 0; i < this._bindings.length; i++) {
            const binding = this._bindings[i];
            if (binding.descriptorType & DESCRIPTOR_DYNAMIC_TYPE) {
                for (let j = 0; j < binding.count; j++) {
                    dynamicBindings.push(binding.binding);
                }
            }
        }

        this._gpuDescriptorSetLayout = {
            bindings: this._bindings,
            dynamicBindings,
            descriptorIndices,
            descriptorCount,
            bindGroupLayout: bindGrpLayout!,
        };

        return true;
    }

    public destroy () {
        this._bindings.length = 0;
    }
}
