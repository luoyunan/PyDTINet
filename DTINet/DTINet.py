from DCA import DCA
from IMC import IMC


class DTINet():
    def __init__(self,
        drug_network_file=None,
        prot_network_file=None,
        drug_sim_nets=None,
        prot_sim_nets=None,
        label_file=None,
        label_mat=None,
        dim_drug=None,
        dim_prot=None,
        imc_dim=None,
        imc_lambda=None,
        imc_solver_type=10,
        imc_max_iter=10,
        imc_threads=4,
        imc_seed=None,
        rwr_rsp=0.5,
        rwr_max_iter=50,
        rwr_epsilon=1e-6,
    ):
        """
        DTINet class.
        Parameters
        ----------
        drug_network_file : str or list of str
            Drug network file(s).
            Every file should be a similarity network with shape (Nd, Nd).
        prot_network_file : str or list of str
            Protein network file(s).
            Every file should be a similarity network with shape (Np, Np).
        drug_sim_nets : list of numpy.ndarray
            Drug similarity network(s).
            drug_network_file will be ignored if drug_sim_nets is provided.
        prot_sim_nets : list of numpy.ndarray
            Protein similarity network(s).
            prot_network_file will be ignored if prot_sim_nets is provided.
        label_file : str
            DTI label file. The file should have a shape of (Nd, Np).
        label_mat : numpy.ndarray
            DTI label 2D array. The array should have a shape of (Nd, Np).
        dim_drug : int
            Dimension of drug embedding.
        dim_prot : int
            Dimension of protein embedding.
        imc_dim : int
            Latent dimension of IMC.
        imc_lambda : float
            Lambda parameter for IMC.
        imc_solver_type : int
            Solver type for IMC. See IMC.py for details.
        imc_max_iter : int
            Maximum number of iterations for IMC.
        imc_threads : int
            Number of threads for IMC.
        imc_seed : int
            Random seed for IMC. (default: None)
        rwr_rsp : float
            Restart probability.
        rwr_max_iter : int
            Maximum number of iterations in RWR.
        rwr_epsilon : float
            Convergence threshold in RWR.
        """
        self.drug_network_file = drug_network_file
        self.prot_network_file = prot_network_file
        self.drug_sim_nets = drug_sim_nets
        self.prot_sim_nets = prot_sim_nets
        self.label_file = label_file
        self.label_mat = label_mat
        self.dim_drug = dim_drug
        self.dim_prot = dim_prot
        self.imc_dim = imc_dim
        self.imc_lambda = imc_lambda
        self.imc_solver_type = imc_solver_type
        self.imc_max_iter = imc_max_iter
        self.imc_threads = imc_threads
        self.imc_seed = imc_seed
        self.rwr_rsp = rwr_rsp
        self.rwr_max_iter = rwr_max_iter
        self.rwr_epsilon = rwr_epsilon

        self._drug_networks = None
        self._prot_networks = None
        self._label = None
        self._drug_feature = None
        self._prot_feature = None
        self._predictor = None


    def _load_network_from_file(self, network_file):
        assert isinstance(network_file, str) or isinstance(network_file, list)
        _files = [network_file] if not isinstance(network_file, list) else network_file
        networks = []
        for f in _files:
            networks.append(np.loadtxt(f))
        return networks


    @property
    def drug_networks(self):
        if self._drug_networks is None:
            if self.drug_sim_nets is not None:
                self._drug_networks = self.drug_sim_nets
            else:
                self._drug_networks = self._load_network_from_file(self.drug_network_file)
        return self._drug_networks


    @property
    def prot_networks(self):
        if self._prot_networks is None:
            if self.prot_sim_nets is not None:
                self._prot_networks = self.prot_sim_nets
            else:
                self._prot_networks = self._load_network_from_file(self.prot_network_file)
        return self._prot_networks


    def _learn_feature(self, networks, dim):
        x = DCA(networks, dim, rsp=self.rwr_rsp,
                max_iter=self.rwr_max_iter, epsilon=self.rwr_epsilon)
        return x

    @property
    def drug_feature(self):
        if self._drug_feature is None:
            self._drug_feature = self._learn_feature(self.drug_networks, self.dim_drug)
        return self._drug_feature

    @property
    def prot_feature(self):
        if self._prot_feature is None:
            self._prot_feature = self._learn_feature(self.prot_networks, self.dim_prot)
        return self._prot_feature

    @property
    def label(self):
        if self._label is None:
            if self.label_mat is not None:
                self._label = self.label_mat
            else:
                self._label = np.loadtxt(self.label_file)
        return self._label

    def train(self):
        self.fit(self.label)

    def fit(self, Y):
        self.predictor = IMC(Y, self.drug_feature, self.prot_feature, self.imc_dim, self.imc_lambda,
                    solver_type=self.imc_solver_type, maxiter=self.imc_max_iter,
                    threads=self.imc_threads, seed=self.imc_seed)

    def predict(self, index):
        """
        Predict the label of a given array of indices.
        Parameters
        ----------
        index : numpy.ndarray
            Array of indices with shape (*, 2).
            The first column is the index of drug and the second column is the index of protein.
        """
        S = self.predictor.predict_mf()
        y_pred = S[index[:, 0], index[:, 1]]
        return y_pred
