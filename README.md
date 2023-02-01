# Learning-Dynamics-from-Coarse-Noisy-Data-with-Scalable-Symbolic-Regression

Abstract
Distilling equations from data can provide insights into physics systems, helping validate theoretical modeling, infer unknown system properties, and explore indeterminate processes. Noisy or downsampled data have been a bottleneck limiting wide applications of symbolic regression, since identified equations are sensitively affected by data statistics. Coarse and noisy data can deteriorate equation reliability by magnifying errors in derivative estimation. While physics-informed surrogate models have been introduced in the literature to reconstruct augmented data that are high-resolution, less affected by noise, and more consistent with underlying physics mechanism ("physics-consistent") for symbolic regression, dominant symbolic regression methods with data augmentation features are reluctant to consider wide function search space. This is due to the optimization burden in nontrivial function search, which is further exacerbated when dynamics exhibit chaotic or rapid oscillation. In this paper, a novel physics-informed equation learning method is proposed to address these issues. Specifically, leveraging a Fourier feature mapping enables a regular and accessible fully-connected neural network (the surrogate model) to learn dynamics with various frequency components. A neural-network-based symbolic model is improved to efficiently represent and separate function combinations in the form of polynomial series. Joint training of the surrogate model and the symbolic model enables "physics-consistent" data augmentation to the original low-quality data and lays the ground for a more reliable equation discovery. The proposed method is demonstrated by numerical and experimental systems parameterized by ordinary differential equations. Compared with baseline methods, such as sparse Bayesian learning and physics-informed neural network with dictionaries, it is found that the proposed method possesses evident competitive edges regarding optimization tractability for scalable function search. 

## Citation
<pre>
@article{chen2023learning,
  title={Learning dynamics from coarse/noisy data with scalable symbolic regression},
  author={Chen, Zhao and Wang, Nan},
  journal={Mechanical Systems and Signal Processing},
  volume={190},
  pages={110147},
  year={2023},
  publisher={Elsevier}
}
</pre>
