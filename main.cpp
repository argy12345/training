#include "dlib/svm.h"
#include "dlib/dnn.h"
#include <ranges>
#include "LightGBM/c_api.h"
#include <vector>
#include <tuple>
typedef dlib::matrix<float> sample;
typedef std::pair<std::vector<sample>, std::vector<float>>& inputs;

std::vector<std::pair<std::vector<float>, int>> loadCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filepath);
    }

    std::vector<std::pair<std::vector<float>, int>> dataset;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;

        // Split by commas
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                int num = std::stoi(token);
                values.push_back(static_cast<float>(num));
            }
        }

        if (values.empty()) continue;

        // Last value = label
        int label = values.back();
        values.pop_back();

        dataset.emplace_back(values, label);
    }

    return dataset;
}


std::pair<std::vector<sample>, std::vector<float>> convert_matrix(std::vector<std::pair<std::vector<float>, int>>& embeddings) {
    std::vector<float> labels;
    auto iters_samples  = embeddings | std::ranges::views::transform([&labels](std::pair<std::vector<float>, int>& input) {
        sample m;
        int c = 0;
        for (auto& embedding: input.first ) {
            m(c)  = (embedding);
            c++; }
        labels.push_back(static_cast<float>(input.second));
        std::pair<std::vector<float>, int> temperary = std::make_pair(std::move(input.first), std::move(input.second));
        return m; });

    std::vector<sample> samples(iters_samples.begin(), iters_samples.end());
    return std::make_pair(samples, labels);
}

void train_gbtree(inputs & data) {
    int ncols = data.first[0].size();
    int nrows = data.first.size();
    std::vector<float> flattened(ncols * nrows);
    auto j = data.first | std::ranges::views::join;
    flattened.assign(std::ranges::begin(j),std::ranges::end(j));

    DatasetHandle input_data;
    LGBM_DatasetCreateFromMat(flattened.data(), C_API_DTYPE_FLOAT64, nrows, ncols,1, nullptr, nullptr,  &input_data  );
    LGBM_DatasetSetField(
     &input_data,
     "label",
     data.second.data(),
     nrows,
     C_API_DTYPE_FLOAT32
 );

    // research params
    const char* params =
        "objective=regression "
        "device=gpu"
        "metric=l2 "
        "learning_rate=0.05 "
        "num_leaves=31 "
        "max_depth=-1 "
        "verbose=-1 "
        "feature_pre_filter=false";

    BoosterHandle booster;
    LGBM_BoosterCreate(input_data, params, &booster);

    int num_iterations = 200; // just train for this many iterations

    for (int iter = 0; iter < num_iterations; ++iter) {
        int finished;
        LGBM_BoosterUpdateOneIter(booster, &finished);
    }

    LGBM_BoosterSaveModel(booster, 0, -1, C_API_FEATURE_IMPORTANCE_GAIN,"model.txt");

}

template <typename SUBNET> using fci =  dlib::relu<dlib::fc<128, SUBNET>>;
template <typename SUBNET> using fc2 = dlib::relu<dlib::fc<64, SUBNET>>;
template <typename SUBNET> using fc3 = dlib::fc<1, SUBNET>;


template <typename SUBNET> using w = dlib::relu<dlib::fc<512, SUBNET>>;
using wider = dlib::loss_binary_log<fci<w<dlib::input<dlib::matrix<float>>>>>;
using v_shallow_net  = dlib::loss_binary_log<fc3<fci<dlib::input<dlib::matrix<float>>>>>;
using shallow_net  = dlib::loss_binary_log<fc3<fc2<fci<dlib::input<dlib::matrix<float>>>>>>;
using log_regression = dlib::loss_binary_log<fc3<dlib::input<dlib::matrix<float>>>>;

// svm types
using kernel_type = dlib::radial_basis_kernel<sample>;
void trian_svm(inputs& data) {
    dlib::svm_c_trainer<kernel_type> trainer;
    trainer.set_kernel(kernel_type(1.0 ));
    trainer.set_c(0.05);

    dlib::decision_function<kernel_type> df = trainer.train(data.first, data.second);

    dlib::serialize("svm_prompt_injection") << df;
}

void train_logistic_reg(inputs& data ) {
    log_regression net;
    dlib::dnn_trainer<log_regression> trainer(net, dlib::sgd());
    trainer.set_learning_rate(0.01); trainer.set_mini_batch_size(32); trainer.set_max_num_epochs(100);
    trainer.train(data.first, data.second);

    dlib::serialize("lr_prompt_injection") << net;

}

template<typename NN >
void train_shallow_ffnn(inputs& data) {
    NN net;
    dlib::dnn_trainer<shallow_net> trainer(net, dlib::sgd());
    trainer.set_learning_rate(0.01); trainer.set_mini_batch_size(32); trainer.set_max_num_epochs(100);
    trainer.train(data.first, data.second);
    dlib::serialize("ffnn_prompt_injection") << net;


}

int main() {
    auto embeddings = loadCSV("sentence_embs.csv");
    auto inputs = convert_matrix(embeddings);

    return 0;
}
