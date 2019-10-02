#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <memory>

torch::Tensor mat2tensor(cv::Mat img)
{
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, CV_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0/255.0);
    float *mat_data = (float*)img.data;

    torch::Tensor tensor = torch::ones({1, 224, 224 , 3});
    float *tensor_data = tensor.data_ptr<float>();

    memcpy(tensor_data, mat_data, 224 * 224 * 3 * sizeof(float));
    tensor = tensor.permute({0,3,1,2});

    return tensor;
}

std::vector<std::string> read_imagenet_names(std::string filepath)
{
    std::vector<std::string> names;

    std::ifstream file(filepath);
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            int delimiter = line.find(':');
            int idx = std::stoi(line.substr(1, delimiter-1));
            std::string name = line.substr(delimiter + 3, line.length() - 9);

            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            names.push_back(name);
        }
        file.close();
    }

    return names;
}

int main()
{
    torch::jit::script::Module module;
    module = torch::jit::load("/tmp/python/model.pt");

    cv::Mat image = cv::imread("/tmp/python/tiger.jpg", CV_LOAD_IMAGE_COLOR);
    torch::Tensor tensor = mat2tensor(image);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);

    torch::Tensor output = module.forward(inputs).toTensor();
    output = output.softmax(1);
    std::tuple<torch::Tensor, torch::Tensor> top_k_output = output.topk(5);
    torch::Tensor probs = std::get<0>(top_k_output);
    torch::Tensor idxs = std::get<1>(top_k_output);

    std::vector<std::string> names = read_imagenet_names("/tmp/python/imagenet_names.txt");
    for (int i=0;i<5;i++)
    {
        float prob = probs[0][i].item<float>();
        int idx = idxs[0][i].item<int>();
        std::string name = names[idx];
        std::cout<<"--class: " + name + " --prob: " + std::to_string(prob)<<std::endl;
    }
}
