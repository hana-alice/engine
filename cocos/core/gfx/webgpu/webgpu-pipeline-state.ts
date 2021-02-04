import { PipelineState, PipelineStateInfo } from '../pipeline-state';
import { IWebGPUGPUPipelineState } from './webgpu-gpu-objects';
import { WebGPURenderPass } from './webgpu-render-pass';
import { WebGPUShader } from './webgpu-shader';
import { DynamicStateFlagBit } from '../define';
import { WebGPUPipelineLayout } from './webgpu-pipeline-layout';

const WebGLPrimitives: GLenum[] = [
    0x0000, // WebGLRenderingContext.POINTS,
    0x0001, // WebGLRenderingContext.LINES,
    0x0003, // WebGLRenderingContext.LINE_STRIP,
    0x0002, // WebGLRenderingContext.LINE_LOOP,
    0x0000, // WebGLRenderingContext.NONE,
    0x0000, // WebGLRenderingContext.NONE,
    0x0000, // WebGLRenderingContext.NONE,
    0x0004, // WebGLRenderingContext.TRIANGLES,
    0x0005, // WebGLRenderingContext.TRIANGLE_STRIP,
    0x0006, // WebGLRenderingContext.TRIANGLE_FAN,
    0x0000, // WebGLRenderingContext.NONE,
    0x0000, // WebGLRenderingContext.NONE,
    0x0000, // WebGLRenderingContext.NONE,
    0x0000, // WebGLRenderingContext.NONE,
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

        this._gpuPipelineState = {
            glPrimitive: WebGLPrimitives[info.primitive],
            gpuShader: (info.shader as WebGPUShader).gpuShader,
            gpuPipelineLayout: (info.pipelineLayout as WebGPUPipelineLayout).gpuPipelineLayout,
            rs: info.rasterizerState,
            dss: info.depthStencilState,
            bs: info.blendState,
            gpuRenderPass: (info.renderPass as WebGPURenderPass).gpuRenderPass,
            dynamicStates,
        };

        return true;
    }

    public destroy () {
        this._gpuPipelineState = null;
    }
}
