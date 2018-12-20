/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_REF_RNN_HPP
#define CPU_REF_RNN_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "../cpu_isa_traits.hpp"
#include "../gemm/os_blas.hpp"

#include "cpu_rnn_pd.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define elemwise_sig(f)                                                      \
    void f(const rnn_utils::rnn_conf_t &rnn,                                 \
            float *ws_gates_, float *states_t_l_,                            \
            float *states_t_lm1_, float *states_tm1_l_,                      \
            float *diff_states_t_l_, float *diff_states_t_lp1_,              \
            float *diff_states_tp1_l_, float *bias_, float *ws_grid_,        \
            float *ws_cell_) const

#define cell_execution_sig(f)                                               \
    void f(const rnn_utils::rnn_conf_t &rnn, float *states_t_l_,            \
            float *diff_states_t_l_, float **w_layer_, float **w_iter_,     \
            float **bias_, float *states_t_lm1_, float *states_tm1_l_,      \
            float *diff_states_t_lp1_, float *diff_states_tp1_l_,           \
            float *diff_w_layer_, float *diff_w_iter_, float *diff_bias_,   \
            float *ws_gates_, float *ws_grid_, float *ws_cell_) const

#define grid_execution_sig(f)                                               \
    void f(const rnn_utils::rnn_conf_t &rnn, float **weights_layer_,        \
            float **weights_states_,                                        \
            float **bias_, float *ws_states_, float *ws_diff_states_,       \
            float *ws_gates_, float *ws_cell_, float *ws_grid_,             \
            float *diff_weights_layer_, float *diff_weights_iter_,          \
            float *diff_bias_) const

#define gemm_sig(f)                                                      \
    void f(const char transA, const char transB, int m, int n, int k,    \
            const float alpha, const float *a_, const int ldA,           \
            const float *b_, const int ldB, const float beta, float *c_, \
            const int ldC) const

#define bias_prepare_sig(f)                                              \
    void f(const rnn_utils::rnn_conf_t &rnn,                             \
            float **bias_, const float *b_, float *scratch_bias_) const

#define bias_finalize_sig(f)                                             \
    void f(const rnn_utils::rnn_conf_t &rnn,                             \
            float **bias_, const float *b_, float *scratch_bias_) const

#define packing_sig(f)                                                         \
    void f(const rnn_utils::rnn_conf_t &rnn, memory_format_t fmt, int OC_size, \
            int IC_size, const int n_parts, const int *gates_per_part,         \
            const size_t *part_weights_pack_size, float **weights_,            \
            const float *w_, float *scratch_weights_, float **bias_,           \
            const float *b_, float *scratch_bias_, bool do_copy) const

template <alg_kind_t alg_kind, prop_kind_t prop_kind>
float activation(float s, float alpha, float cliping, float dd);

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public cpu_primitive_t {
    using class_name = _ref_rnn_common_t<aprop>;
    typedef elemwise_sig((class_name::*elemwise_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));

    typedef gemm_sig((class_name::*gemm_t));
    typedef bias_prepare_sig((class_name::*bias_prepare_t));
    typedef bias_finalize_sig((class_name::*bias_finalize_t));
    typedef packing_sig((class_name::*packing_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_pd)
            : base_pd_t(engine, adesc, attr, hint_pd) {}

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace memory_format;
            using namespace rnn_utils;
            assert(this->engine()->kind() == engine_kind::cpu);
            const alg_kind_t cell_kind = this->desc()->cell_desc.cell_kind;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                               alg_kind::gru_linear_before_reset)
                    && IMPLICATION(aprop == prop_kind::forward,
                               one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference))
                    && IMPLICATION(aprop == backward,
                               one_of(this->desc()->prop_kind, backward))
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok)
                return status::unimplemented;

            init_conf(rnn_, *this->desc(), this->src_pd(0), this->src_pd(1),
                    this->weights_pd(0), this->weights_pd(1), this->dst_pd(0));

            // Set weights descriptors to desired format
            memory_desc_t weights_layer_md = *(this->weights_layer_pd_.desc());
            CHECK(set_expected_desc(rnn_, weights_layer_md, false));
            cpu_memory_t::pd_t new_weights_layer_pd(
                    this->engine_, &weights_layer_md);
            if (this->weights_layer_pd_.desc()->format == any) {
                this->weights_layer_pd_ = new_weights_layer_pd;
            } else if (!this->weights_layer_pd_.is_equal(&new_weights_layer_pd))
                 return status::unimplemented;

            memory_desc_t weights_iter_md = *(this->weights_iter_pd_.desc());
            CHECK(set_expected_desc(rnn_, weights_iter_md, true));
            cpu_memory_t::pd_t new_weights_iter_pd(
                    this->engine_, &weights_iter_md);
            if (this->weights_iter_pd_.desc()->format == any) {
                this->weights_iter_pd_ = new_weights_iter_pd;
            } else if (!this->weights_iter_pd_.is_equal(&new_weights_iter_pd))
                return status::unimplemented;

            CHECK(this->check_layout_consistency());

            set_conf(rnn_, *this->desc(), this->weights_pd(0),
                    this->weights_pd(1), this->diff_weights_pd(0),
                    this->diff_weights_pd(1));

            size_t scratchpad_sz{0}, ws_sz{0};
            get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

            // initialize the workspace_pd if needed
            if (rnn_.is_training) {
                dims_t ws_dims = {(int)ws_sz};
                memory_desc_t ws_d;
                mkldnn_memory_desc_init(&ws_d, 1, ws_dims, data_type::f32, x);
                this->ws_pd_ = cpu_memory_t::pd_t(this->engine(), &ws_d);
            }

            init_scratchpad(scratchpad_sz);

            return status::success;
        }

        rnn_utils::rnn_conf_t rnn_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, sizeof(float) * scratchpad_sz, 4096);

            int max_nparts = this->cell_kind() == alg_kind::vanilla_gru ? 2 : 1;
            int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.book(key_rnn_ptrs_wei_layer,
                    sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_ptrs_wei_iter,
                    sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_ptrs_bia,
                    sizeof(float *) * ptr_wei_sz);
        }
    };

    _ref_rnn_common_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        auto set_pack_funcs = [](bool packed_gemm, gemm_t &g, bool pack_w,
                bool copy_w, bool already_packed, packing_t &p) {
            g = packed_gemm ? &class_name::packed_gemm : &class_name::gemm;
            if (pack_w)
                p = &class_name::pack_weights;
            else if (copy_w)
                p = &class_name::copy_weights;
            else
                p = already_packed
                    ? &class_name::assign_packed_weights
                    : &class_name::assign_weights;
        };
        set_pack_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                pd()->rnn_.pack_weights_iter, pd()->rnn_.copy_weights_iter,
                pd()->rnn_.weights_iter_is_packed, weights_iter_pack_func);

        set_pack_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                pd()->rnn_.pack_weights_layer, pd()->rnn_.copy_weights_layer,
                pd()->rnn_.weights_layer_is_packed, weights_layer_pack_func);

        switch (pd()->cell_kind()) {
        case alg_kind::vanilla_lstm:
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::lstm_elemwise;
            break;
        case alg_kind::vanilla_rnn: // @todo switch on cell kind
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::rnn_elemwise;
            switch (pd()->activation_kind()) {
            case alg_kind::eltwise_relu:
                activation_func = &activation<alg_kind::eltwise_relu, aprop>;
                break;
            case alg_kind::eltwise_tanh:
                activation_func = &activation<alg_kind::eltwise_tanh, aprop>;
                break;
            case alg_kind::eltwise_logistic:
                activation_func = &activation<alg_kind::eltwise_logistic, aprop>;
                break;
            default: break;
            }
            break;
        case alg_kind::vanilla_gru:
            cell_func = &class_name::cell_execution_gru;
            break;
        case alg_kind::gru_linear_before_reset:
            cell_func = &class_name::cell_execution_gru_lbr;
            elemwise_func = &class_name::gru_lbr_elemwise;
            break;
        default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_states_offset_,
                ws_diff_states_offset_, ws_grid_comp_offset_,
                ws_cell_comp_offset_, ws_weights_layer_offset_,
                ws_weights_iter_offset_, ws_bias_offset_, ws_diff_weights_layer_offset_,
                ws_diff_weights_iter_offset_, scratchpad_size, workspace_size);
    }

    ~_ref_rnn_common_t() {}

    // typedef typename prec_traits::type data_t;

    virtual void execute(event_t *e) const {
        execute_();
        e->set_state(event_t::ready);
    }

private:
    void execute_() const;
    grid_execution_sig(linear_execution);
    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);
    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(gru_lbr_elemwise);
    gemm_sig(gemm);
    gemm_sig(packed_gemm);
    bias_prepare_sig(bias_prepare);
    bias_finalize_sig(bias_finalize);
    packing_sig(pack_weights);
    packing_sig(assign_weights);
    packing_sig(copy_weights);
    packing_sig(assign_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn, float *ws_states_,
            float *ws_diff_states_, const float *xt_,
            const float *diff_dst_layer) const;
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn, float *ws_states_,
            float *ws_diff_states_, const float *firstit_states_,
            const float *diff_dst_iter) const;
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn, float *dst_layer_,
            float *diff_src_layer, const float *ws_states_,
            const float *ws_diff_states_) const;
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn, float *dst_iter_,
            float *diff_src_iter, const float *ws_states_,
            const float *ws_diff_states_) const;
    void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
            const float *ws_gates_, float *diff_bias_) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    size_t ws_gates_offset_;
    size_t ws_states_offset_;
    size_t ws_weights_layer_offset_;
    size_t ws_weights_iter_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_offset_;
    size_t ws_diff_weights_layer_offset_;
    size_t ws_diff_weights_iter_offset_;
    size_t ws_grid_comp_offset_;
    size_t ws_cell_comp_offset_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    packing_t weights_layer_pack_func;
    packing_t weights_iter_pack_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
    elemwise_f elemwise_func;

};

using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;

}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
