/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain src copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "cpu_isa_traits.hpp"
#include "dnnl.hpp"

#include <vector>

namespace dnnl {

template<typename T>
static void run_matmul(const std::vector<T>& x, 
                       const std::vector<T>& y,
                       float scale_x,
                       float scale_y, 
                       std::vector<float>& out) {

    dnnl::engine engine = dnnl::engine(get_test_engine_kind(), 0);
    auto data_type = std::is_same<T, float>::value ? memory::data_type::f32 
                                                   : memory::data_type::s8;
    const memory::dim batch_size = 1;
    const memory::dim M = 2;
    const memory::dim N = 2;
    const memory::dim K = 2;

    memory::dims src_dims = {batch_size, M, K};
    memory::dims weights_dims = {batch_size, K, N};
    memory::dims dst_dims = {batch_size, M, N};

    auto src_md = memory::desc(src_dims, data_type, memory::format_tag::abc);
    auto weights_md =
        memory::desc(weights_dims, data_type, memory::format_tag::abc);
    auto src_mem = dnnl::memory(src_md, engine, (void*)x.data());
    auto weights_mem =
        dnnl::memory(weights_md, engine, (void*)y.data());

    memory::desc dst_md =
        memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::abc);
    dnnl::memory dst_mem = dnnl::memory(
        dst_md, engine, out.data());

    float scale_out = 1.0f; // fp32 is forced here.
    float out_shift_scale = scale_out / (scale_x * scale_y);
    float alpha = 1.0f;
    float final_scale_out = out_shift_scale * alpha;

    dnnl::primitive_attr attr;
    attr.set_output_scales(/* mask */ 0, {final_scale_out});

    auto matmul_d = dnnl::matmul::desc(src_md, weights_md, dst_md);
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine);
    auto matmul_prim = dnnl::matmul(matmul_pd);

    dnnl::stream stream(engine);
    matmul_prim.execute(stream, {{DNNL_ARG_SRC, src_mem},
                                 {DNNL_ARG_WEIGHTS, weights_mem},
                                 {DNNL_ARG_DST, dst_mem},
                                 });
    stream.wait();
}

TEST(Test, test) {
    dnnl::engine engine = dnnl::engine(get_test_engine_kind(), 0);
    const int num_elements = 4;
    std::vector<float> x_data = {1.f, 1.f, 0.f, 1.f};
    std::vector<float> y_data = {1.f, 1.f, 0.f, 1.f};

    const float max_x = *std::max_element(x_data.begin(), x_data.end());
    const float max_y = *std::max_element(y_data.begin(), y_data.end());
    const float scale_x = 127.f / max_x;
    const float scale_y = 127.f / max_y;

    std::vector<int8_t> x_quantized_data = {127, 127, 0, 127};
    std::vector<int8_t> y_quantized_data = {127, 127, 0, 127};

    std::vector<float> out_int8_data(num_elements, 0);
    std::vector<float> out_float_data(num_elements, 0);
    run_matmul(x_data, y_data, 1.0f, 1.0f, out_float_data);
    run_matmul(x_quantized_data, y_quantized_data, scale_x, scale_y, out_int8_data);

    std::cout << "dnnl int8 output: " << std::endl;
    for(int i = 0; i < num_elements; i++) {
        std::cout << std::to_string(out_int8_data[i]) << ' ';
    }
    std::cout << std::endl;

    std::cout << "dnnl fp32 output" << std::endl;
    for(int i = 0; i < num_elements; i++) {
        std::cout << std::to_string(out_float_data[i]) << ' ';
    }
    std::cout << std::endl;

    for(int i = 0; i < num_elements; i++) {
        EXPECT_NEAR(out_int8_data[i], out_float_data[i], 0.01f);
    }
}
} // namespace dnnl
