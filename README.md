Code for our paper, "Covariance and Principal Component Analysis on Riemannian Manifolds and Graphs," by Luiz Hartmann, Wenwen Li, and Washington Mio.

The paper is available on arXiv at: https://arxiv.org/abs/2609.15753

The experiment notebooks were prepared in Google Colab and use the standalone Python modules included in this repository.

- `Torus.ipynb` and `Two_clusters.ipynb` require `riepca_euclidean.py`;
- `riepca_euclidean.py`. `Sphere.ipynb`, `EEG_(2_tasks_1_subject).ipynb`, and `EEG_cross_session_PC3_PC4_SVM.ipynb` require `riepca_manifold.py`.

Download each notebook and its required module (`.py` file) from the same repository. To run a notebook in Google Colab, download the required module, open the notebook in Colab, and upload the module to Colab. Place it directly in `/content`, keeping its original filename, before running the notebook cells from top to bottom. Some notebooks prompt for this upload automatically. Uploaded files are temporary and must be uploaded again if the runtime is deleted or recreated.

For local Jupyter execution, place each notebook and its required module in the same directory, start the notebook from that directory, and install its Python dependencies. The final Colab-specific download cell in `Sphere.ipynb` can be skipped locally.
