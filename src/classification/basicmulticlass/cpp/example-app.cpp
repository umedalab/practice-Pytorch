// Minimal code
//#include <torch/torch.h>
//#include <iostream>
//
//int main() {
//  torch::Tensor tensor = torch::rand({2, 3});
//  std::cout << tensor << std::endl;
//}

#include <iostream>

#ifdef UNICODE // https://github.com/pytorch/pytorch/issues/27568
#define UNICODE_TMP UNICODE
#undef UNICODE
#endif
#include <torch/torch.h>
#include <torch/script.h>
#ifdef UNICODE_TMP
#define UNICODE UNICODE_TMP
#undef UNICODE_TMP
#endif

#include <opencv2/opencv.hpp>

#include <memory>
#include <windows.h>
#include <time.h>

int main()
{
	std::cout << "Load module" << std::endl;
	// load the model
	try
	{
		int img_width = 128, img_height = 128;
		//auto module = torch::jit::load("D:\\workspace\\programs\\ThirdPartyProg\\CNN\\Custom-CNN-based-Image-Classification-in-PyTorch-master\\traced_model.pt");
		auto module = torch::jit::load("../../python/traced_model.pt");

		std::cout << "Model was loaded." << std::endl;

		// send model to CUDA
		module.to(at::kCUDA);
		module.eval();
		// define the base tensor
		at::Tensor input = torch::ones({ 1, img_width, img_height, 3 });
		// manually modify a tensor parameter
		//input[0][0][0][0] = 0.12345678901234;
		//std::cout << input << std::endl;

		cv::Mat frame, resized_frame, rgb_frame;

		int key;
		std::string fname = "../../../../../data/classification/classes4/test/car/010.jpg";
		frame = cv::imread(fname);
		if (!frame.empty()) {
			std::cout << "[+] open:" << fname << std::endl;
		} else {
			std::cout << "[-] open:" << fname << std::endl;
			return 0;
		}
		cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
		cv::resize(frame, resized_frame, cv::Size(img_width, img_height));
		clock_t time1 = clock();
		//auto frame_tensor = subsCvMat2Tensor(input, resized_frame);

		auto frame_tensor = torch::from_blob(resized_frame.data, { 1, resized_frame.rows, resized_frame.cols, 3 }, at::kByte);
		frame_tensor = frame_tensor.to(at::kFloat).div(255.0).clamp(0.0, 1.0);
		frame_tensor = frame_tensor.permute({ 0, 3, 1, 2 });
		//std::cout << frame_tensor << std::endl;
		clock_t time2 = clock();
		std::cout << "transpose_time:" << (double)(time2 - time1) / CLOCKS_PER_SEC << std::endl;

		// Send to CUDA
		auto input_cuda = frame_tensor.to(at::kCUDA);

		// Model‚Ö“ü—Í
		auto output_cuda = module.forward({ input_cuda }).toTensor();
		// Send to CPU
		auto output = output_cuda.to(at::kCPU).detach();
		std::cout << "output: }}}" << output << "{{{" << std::endl;
		std::cout << ">> " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

		clock_t end = clock();
		std::cout << "model_time:" << (double)(end - time2) / CLOCKS_PER_SEC << std::endl;

		cv::imshow("frame", frame);
		key = cv::waitKey(0);
		cv::destroyAllWindows();
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}
	catch (std::exception e) {
		std::cout << "e: " << e.what() << std::endl;
		return -1;
	}
	catch (...) {
		// this executes if f() throws std::string or int or any other unrelated type
		std::cout << "exception non managed" << std::endl;
	}
	return 0;
}

