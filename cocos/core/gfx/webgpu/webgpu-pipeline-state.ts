import { PipelineState, PipelineStateInfo } from '../pipeline-state';
import { IWebGPUGPUPipelineState } from './webgpu-gpu-objects';
import { WebGPURenderPass } from './webgpu-render-pass';
import { WebGPUShader } from './webgpu-shader';
import { CullMode, DynamicStateFlagBit, ShaderStageFlagBit } from '../define';
import { WebGPUPipelineLayout } from './webgpu-pipeline-layout';
import { WebGPUDevice } from './webgpu-device';
import { GFXFormatToWebGLInternalFormat, WebGPUBlendFactors, WebGPUBlendOps, WebGPUCompereFunc, WebGPUStencilOp } from './webgpu-commands';

const WebPUPrimitives: GPUPrimitiveTopology[] = [
    'point-list',
    'line-list',
    'line-strip',
    'line-strip',   // no line_loop in webgpu
    'line-list',
    'line-strip',
    'line-list',
    'triangle-list',
    'triangle-strip',
    'triangle-strip',
    'triangle-list',
    'triangle-strip',
    'triangle-strip',
    'triangle-strip',    // no quad
];

export class WebGPUPipelineState extends PipelineState {
    get gpuPipelineState (): IWebGPUGPUPipelineState {
        return  this._gpuPipelineState!;
    }

    private _gpuPipelineState: IWebGPUGPUPipelineState | null = null;

    public initialize (info: PipelineStateInfo): boolean {
        this._primitive = info.primitive;
        this._shader = info.shader;
        this._pipelineLayout = info.pipelineLayout;
        this._rs = info.rasterizerState;
        this._dss = info.depthStencilState;
        this._bs = info.blendState;
        this._is = info.inputState;
        this._renderPass = info.renderPass;
        this._dynamicStates = info.dynamicStates;

        const dynamicStates: DynamicStateFlagBit[] = [];
        for (let i = 0; i < 31; i++) {
            if (this._dynamicStates & (1 << i)) {
                dynamicStates.push(1 << i);
            }
        }

        const renderPplDesc = {} as GPURenderPipelineDescriptor;

        // pipelinelayout
        const nativePipelineLayout = (this._pipelineLayout as WebGPUPipelineLayout).gpuPipelineLayout.nativePipelineLayout;
        renderPplDesc.layout = nativePipelineLayout;

        // shadestage
        const shaderStages = (this._shader as WebGPUShader).gpuShader.gpuStages;
        for (let i = 0; i < shaderStages.length; i++) {
            if (shaderStages[i].type === ShaderStageFlagBit.VERTEX) { renderPplDesc.vertexStage = shaderStages[i].glShader!; }
            if (shaderStages[i].type === ShaderStageFlagBit.FRAGMENT) { renderPplDesc.fragmentStage = shaderStages[i].glShader!; }
        }

        // primitive
        renderPplDesc.primitiveTopology = WebPUPrimitives[info.primitive];

        // rs
        renderPplDesc.rasterizationState = {
            frontFace: this._rs.isFrontFaceCCW ? 'ccw' : 'cw',
            cullMode: this._rs.cullMode === CullMode.NONE ? 'none' : (this._rs.cullMode === CullMode.FRONT) ? 'front' : 'back',
            clampDepth: this._rs.isDepthClip,
            depthBias: this._rs.depthBias,
            depthBiasSlopeScale: this._rs.depthBiasSlop,
            depthBiasClamp: this._rs.depthBiasClamp,
        };

        // colorstates
        const colorAttachments = this._renderPass.colorAttachments;
        const colorDescs: GPUColorStateDescriptor[] = [];
        for (let i = 0; i < colorAttachments.length; i++) {
            colorDescs.push({
                format: GFXFormatToWebGLInternalFormat(colorAttachments[i].format),
                alphaBlend: {
                    dstFactor: WebGPUBlendFactors[this._bs.targets[i].blendDstAlpha],
                    operation: WebGPUBlendOps[this._bs.targets[i].blendAlphaEq],
                    srcFactor: WebGPUBlendFactors[this._bs.targets[i].blendSrcAlpha],
                },
                colorBlend: {
                    dstFactor: WebGPUBlendFactors[this._bs.targets[i].blendDst],
                    operation: WebGPUBlendOps[this._bs.targets[i].blendEq],
                    srcFactor: WebGPUBlendFactors[this._bs.targets[i].blendSrc],
                },
                writeMask: this._bs.targets[i].blendColorMask,
            });
        }
        renderPplDesc.colorStates = colorDescs;

        // depthstencil states
        if (this._renderPass.depthStencilAttachment) {
            const dssDesc = {} as GPUDepthStencilStateDescriptor;
            dssDesc.format = GFXFormatToWebGLInternalFormat(this._renderPass.depthStencilAttachment.format);
            dssDesc.depthWriteEnabled = this._dss.depthWrite;
            dssDesc.depthCompare = WebGPUCompereFunc[this._dss.depthFunc];
            let stencilReadMask = 0x0;
            let stencilWriteMask = 0x0;

            if (this._dss.stencilTestFront) {
                dssDesc.stencilFront = {
                    compare: WebGPUCompereFunc[this._dss.stencilFuncFront],
                    depthFailOp: WebGPUStencilOp[this._dss.stencilZFailOpFront],
                    passOp: WebGPUStencilOp[this._dss.stencilPassOpFront],
                    failOp: WebGPUStencilOp[this._dss.stencilFailOpFront],
                };
                stencilReadMask |= this._dss.stencilReadMaskFront;
                stencilWriteMask |= this._dss.stencilWriteMaskFront;
            }
            if (this._dss.stencilTestBack) {
                dssDesc.stencilBack = {
                    compare: WebGPUCompereFunc[this._dss.stencilFuncBack],
                    depthFailOp: WebGPUStencilOp[this._dss.stencilZFailOpBack],
                    passOp: WebGPUStencilOp[this._dss.stencilPassOpBack],
                    failOp: WebGPUStencilOp[this._dss.stencilFailOpBack],
                };
                stencilReadMask |= this._dss.stencilReadMaskBack;
                stencilWriteMask |= this._dss.stencilWriteMaskBack;
            }
            dssDesc.stencilReadMask = stencilReadMask;
            dssDesc.stencilWriteMask = stencilWriteMask;
        }

        // -------optional-------
        // renderPplDesc.vertexState = {};
        // renderPplDesc.sampleCount = 0;
        // renderPplDesc.sampleMask = 0;
        // renderPplDesc.alphaToCoverageEnabled = false;

        const nativeDevice = (this._device as WebGPUDevice).nativeDevice();
        const nativePipeline = nativeDevice?.createRenderPipeline(renderPplDesc);

        this._gpuPipelineState = {
            glPrimitive: WebPUPrimitives[info.primitive],
            gpuShader: (info.shader as WebGPUShader).gpuShader,
            gpuPipelineLayout: (info.pipelineLayout as WebGPUPipelineLayout).gpuPipelineLayout,
            rs: info.rasterizerState,
            dss: info.depthStencilState,
            bs: info.blendState,
            gpuRenderPass: (info.renderPass as WebGPURenderPass).gpuRenderPass,
            dynamicStates,
            nativePipeline,
        };

        return true;
    }

    public destroy () {
        this._gpuPipelineState = null;
    }
}
