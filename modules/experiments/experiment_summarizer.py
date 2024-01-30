class ExperimentSummerizer:
    def __init__(self, experiment_saved_path):
        self.saved_model_path = experiment_saved_path 
        self.experiment_labels = []
        self.test_mpjpe_list = []
        self.test_pjpe_list = []
        self.test_activity_mpjpe_list = []
        self.test_activity_macro_mpjpe_list = []
        self.avg_mpjpe = None
        self.avg_activity_macro_mpjpe = None
        self.avg_pjpe = None
        self.avg_activity_mpjpe = None


    def add_result(self, test_mpjpe, test_pjpe, test_activity_mpjpe, test_activity_macro_mpjpe):
        self.test_mpjpe_list.append(test_mpjpe)
        self.test_pjpe_list.append(test_pjpe)
        self.test_activity_mpjpe_list.append(test_activity_mpjpe)
        self.test_activity_macro_mpjpe_list.append(test_activity_macro_mpjpe)

    def calculate(self):
        pass

    def print_raw_result(self):
        pass

    def print_summarize_result(self):
        pass
