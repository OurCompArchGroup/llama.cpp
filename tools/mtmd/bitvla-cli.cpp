#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static constexpr llama_token BITVLA_STOP_TOKEN = 128001;
static constexpr llama_token BITVLA_PROPRIO_TOKEN = 128011;

struct bitvla_options {
    std::vector<float> proprio;
    std::string stats_path;
    std::string stats_key;
};

struct normalization_stats {
    std::vector<float> action_low;
    std::vector<float> action_high;
    std::vector<bool> action_mask;
    std::vector<float> proprio_low;
    std::vector<float> proprio_high;
};

static void print_usage(int /*argc*/, char ** argv) {
    LOG(
        "BitVLA action prediction\n\n"
        "Usage: %s -m MODEL --mmproj MMPROJ --image IMAGE[,IMAGE...] "
        "-p INSTRUCTION --proprio V0,...,V7 [options]\n\n"
        "BitVLA options:\n"
        "  --image FILES       comma-separated images in model input order\n"
        "  --proprio VALUES    comma-separated state vector\n"
        "  --stats FILE        dataset_statistics.json; normalizes state and unnormalizes actions\n"
        "  --stats-key NAME    dataset key in the statistics file (auto-selected when unique)\n",
        argv[0]);
}

static std::vector<float> parse_floats(const std::string & value) {
    std::vector<float> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        size_t used = 0;
        const float parsed = std::stof(item, &used);
        if (used != item.size() || !std::isfinite(parsed)) {
            throw std::invalid_argument("invalid --proprio value: " + item);
        }
        result.push_back(parsed);
    }
    return result;
}

static std::vector<char *> parse_bitvla_args(
        int argc,
        char ** argv,
        bitvla_options & options) {
    std::vector<char *> common_argv = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&]() -> const char * {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[i];
        };

        if (arg == "--proprio") {
            options.proprio = parse_floats(require_value());
        } else if (arg == "--stats") {
            options.stats_path = require_value();
        } else if (arg == "--stats-key") {
            options.stats_key = require_value();
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    return common_argv;
}

static std::vector<float> json_floats(const json & value, const char * name) {
    if (!value.is_array()) {
        throw std::runtime_error(std::string("statistics field is not an array: ") + name);
    }
    return value.get<std::vector<float>>();
}

static normalization_stats load_stats(
        const std::string & path,
        const std::string & requested_key,
        int action_dim,
        int proprio_dim) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open statistics file: " + path);
    }

    json root;
    input >> root;
    if (!root.is_object() || root.empty()) {
        throw std::runtime_error("statistics file has no dataset entries: " + path);
    }

    std::string key = requested_key;
    if (key.empty()) {
        if (root.size() != 1) {
            throw std::runtime_error("--stats-key is required when the statistics file has multiple datasets");
        }
        key = root.begin().key();
    }
    if (!root.contains(key)) {
        throw std::runtime_error("statistics key not found: " + key);
    }

    const auto & dataset = root.at(key);
    const auto & action = dataset.at("action");
    const auto & proprio = dataset.at("proprio");
    normalization_stats stats;
    stats.action_low = json_floats(action.at("q01"), "action.q01");
    stats.action_high = json_floats(action.at("q99"), "action.q99");
    stats.proprio_low = json_floats(proprio.at("q01"), "proprio.q01");
    stats.proprio_high = json_floats(proprio.at("q99"), "proprio.q99");
    stats.action_mask.assign(action_dim, true);
    if (action.contains("mask")) {
        stats.action_mask = action.at("mask").get<std::vector<bool>>();
    }

    if ((int) stats.action_low.size() != action_dim ||
            (int) stats.action_high.size() != action_dim ||
            (int) stats.action_mask.size() != action_dim ||
            (int) stats.proprio_low.size() != proprio_dim ||
            (int) stats.proprio_high.size() != proprio_dim) {
        throw std::runtime_error("statistics dimensions do not match the BitVLA action metadata");
    }
    return stats;
}

static std::vector<float> normalize_proprio(
        const std::vector<float> & proprio,
        const normalization_stats & stats) {
    std::vector<float> result(proprio.size());
    for (size_t i = 0; i < proprio.size(); ++i) {
        const float range = stats.proprio_high[i] - stats.proprio_low[i] + 1e-8f;
        result[i] = std::clamp(2.0f * (proprio[i] - stats.proprio_low[i]) / range - 1.0f, -1.0f, 1.0f);
    }
    return result;
}

static std::vector<float> unnormalize_actions(
        const std::vector<float> & actions,
        const normalization_stats & stats,
        int action_dim) {
    std::vector<float> result(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        const int dim = i % action_dim;
        result[i] = stats.action_mask[dim]
            ? 0.5f * (actions[i] + 1.0f) *
                (stats.action_high[dim] - stats.action_low[dim] + 1e-8f) + stats.action_low[dim]
            : actions[i];
    }
    return result;
}

static std::string make_prompt(std::string instruction, size_t n_images) {
    instruction.erase(instruction.begin(), std::find_if(instruction.begin(), instruction.end(), [](unsigned char c) {
        return !std::isspace(c);
    }));
    instruction.erase(std::find_if(instruction.rbegin(), instruction.rend(), [](unsigned char c) {
        return !std::isspace(c);
    }).base(), instruction.end());
    std::transform(instruction.begin(), instruction.end(), instruction.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (!instruction.empty() && instruction.back() == '?') {
        instruction.pop_back();
    }

    std::string media_markers;
    for (size_t i = 0; i < n_images; ++i) {
        media_markers += mtmd_default_marker();
    }

    return std::string(
        "System: A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.<|eot_id|>"
        "User: ") + media_markers +
        "<proprio_pad>What action should the robot take to " + instruction +
        "?<|eot_id|>Assistant: ";
}

static json action_matrix(const std::vector<float> & values, int chunk, int dim) {
    json result = json::array();
    for (int i = 0; i < chunk; ++i) {
        json row = json::array();
        for (int j = 0; j < dim; ++j) {
            row.push_back(values[(size_t) i * dim + j]);
        }
        result.push_back(std::move(row));
    }
    return result;
}

static bool check_finite(const char * name, const float * values, size_t count) {
    size_t n_invalid = 0;
    float min_value = INFINITY;
    float max_value = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            ++n_invalid;
        } else {
            min_value = std::min(min_value, values[i]);
            max_value = std::max(max_value, values[i]);
        }
    }
    if (n_invalid != 0) {
        LOG_ERR("%s contains %zu non-finite values out of %zu\n", name, n_invalid, count);
        return false;
    }
    LOG_INF("%s range: [%g, %g]\n", name, min_value, max_value);
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    try {
        bitvla_options options;
        auto common_argv = parse_bitvla_args(argc, argv, options);

        common_params params;
        if (!common_params_parse((int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_MTMD, print_usage)) {
            return 1;
        }
        if (params.model.path.empty() || params.mmproj.path.empty() || params.image.empty() ||
                params.prompt.empty() || options.proprio.empty()) {
            print_usage(argc, argv);
            LOG_ERR("model, mmproj, at least one image, instruction, and proprio state are required\n");
            return 1;
        }

        params.embedding = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        common_init();
        mtmd_helper_log_set(common_log_default_callback, nullptr);

        common_init_result_ptr llama_init = common_init_from_params(params);
        llama_model * model = llama_init->model();
        llama_context * lctx = llama_init->context();
        if (!model || !lctx) {
            return 1;
        }

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        base_callback_data cb_data;
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval<false>;
        }
        mtmd::context_ptr mtmd_ctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!mtmd_ctx) {
            LOG_ERR("failed to load BitVLA multimodal model\n");
            return 1;
        }

        mtmd_action_info info;
        if (!mtmd_get_action_info(mtmd_ctx.get(), &info)) {
            LOG_ERR("mmproj does not contain a supported BitVLA action head\n");
            return 1;
        }
        if (info.llm_dim != llama_model_n_embd_inp(model) ||
                (int) options.proprio.size() != info.proprio_dim) {
            LOG_ERR("BitVLA model or proprio dimensions do not match\n");
            return 1;
        }

        normalization_stats stats;
        const bool use_stats = !options.stats_path.empty();
        if (use_stats) {
            stats = load_stats(options.stats_path, options.stats_key, info.action_dim, info.proprio_dim);
            options.proprio = normalize_proprio(options.proprio, stats);
        }

        std::vector<float> proprio_embedding(info.llm_dim);
        if (mtmd_project_proprio(mtmd_ctx.get(), options.proprio.data(), proprio_embedding.data()) != 0) {
            return 1;
        }
        if (!check_finite("proprio embedding", proprio_embedding.data(), proprio_embedding.size())) {
            return 1;
        }

        mtmd::bitmaps bitmaps;
        bitmaps.entries.reserve(params.image.size());
        for (const std::string & image_path : params.image) {
            bitmaps.entries.emplace_back(mtmd_helper_bitmap_init_from_file(mtmd_ctx.get(), image_path.c_str()));
            if (!bitmaps.entries.back().ptr) {
                return 1;
            }
        }
        auto bitmap_ptrs = bitmaps.c_ptr();
        const std::string prompt = make_prompt(params.prompt, bitmap_ptrs.size());
        mtmd_input_text text = {
            /*.text          =*/ prompt.c_str(),
            /*.add_special   =*/ true,
            /*.parse_special =*/ true,
        };
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        if (mtmd_tokenize(mtmd_ctx.get(), chunks.ptr.get(), &text, bitmap_ptrs.data(), bitmap_ptrs.size()) != 0) {
            return 1;
        }

        const int n_embd = info.llm_dim;
        const int n_action_tokens = info.action_chunk * info.action_dim;
        std::vector<float> input_embeddings;
        int n_image_tokens = 0;
        int n_proprio_tokens = 0;
        auto append_embedding = [&](const float * embedding) {
            input_embeddings.insert(input_embeddings.end(), embedding, embedding + n_embd);
        };

        std::vector<float> token_embedding(n_embd);
        for (size_t i = 0; i < chunks.size(); ++i) {
            const mtmd_input_chunk * chunk = chunks[i];
            const auto type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                for (size_t j = 0; j < n_tokens; ++j) {
                    if (tokens[j] == BITVLA_PROPRIO_TOKEN) {
                        append_embedding(proprio_embedding.data());
                        ++n_proprio_tokens;
                    } else {
                        if (llama_model_get_token_embedding(model, tokens[j], token_embedding.data()) != 0) {
                            LOG_ERR("failed to read embedding for token %d\n", tokens[j]);
                            return 1;
                        }
                        append_embedding(token_embedding.data());
                    }
                }
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(mtmd_ctx.get(), chunk) != 0) {
                    return 1;
                }
                const int n_tokens = (int) mtmd_input_chunk_get_n_tokens(chunk);
                const float * image_embeddings = mtmd_get_output_embd(mtmd_ctx.get());
                if (!check_finite("image embeddings", image_embeddings, (size_t) n_tokens * n_embd)) {
                    return 1;
                }
                input_embeddings.insert(input_embeddings.end(), image_embeddings,
                    image_embeddings + (size_t) n_tokens * n_embd);
                n_image_tokens += n_tokens;
            } else {
                LOG_ERR("BitVLA only supports image input\n");
                return 1;
            }
        }

        if (n_image_tokens == 0 || n_proprio_tokens != 1) {
            LOG_ERR("expected image embeddings and exactly one proprio placeholder\n");
            return 1;
        }

        if (const char * stop = std::getenv("BITVLA_STOP_AFTER_VISION")) {
            if (!(stop[0] == '0' && stop[1] == '\0')) {
                LOG_INF("BITVLA_STOP_AFTER_VISION: skipping language model after %d image tokens\n",
                        n_image_tokens);
                return 0;
            }
        }

        const int action_start = (int) input_embeddings.size() / n_embd;
        input_embeddings.resize(input_embeddings.size() + (size_t) n_action_tokens * n_embd, 0.0f);
        if (llama_model_get_token_embedding(model, BITVLA_STOP_TOKEN, token_embedding.data()) != 0) {
            LOG_ERR("failed to read the BitVLA stop-token embedding\n");
            return 1;
        }
        append_embedding(token_embedding.data());
        const int n_tokens = (int) input_embeddings.size() / n_embd;
        if (!check_finite("language-model input", input_embeddings.data(), input_embeddings.size())) {
            return 1;
        }

        if (n_tokens > (int) llama_n_ctx(lctx) || n_tokens > (int) llama_n_batch(lctx) ||
                n_tokens > (int) llama_n_ubatch(lctx)) {
            LOG_ERR("BitVLA sequence has %d tokens, but context/batch/ubatch are %u/%u/%u\n",
                n_tokens, llama_n_ctx(lctx), llama_n_batch(lctx), llama_n_ubatch(lctx));
            return 1;
        }

        llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
        batch.n_tokens = n_tokens;
        std::copy(input_embeddings.begin(), input_embeddings.end(), batch.embd);
        for (int i = 0; i < n_tokens; ++i) {
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i >= action_start && i < action_start + n_action_tokens;
        }

        llama_memory_clear(llama_get_memory(lctx), true);
        llama_set_embeddings(lctx, true);
        llama_set_causal_attn(lctx, false);
        const int decode_result = llama_decode(lctx, batch);
        llama_set_causal_attn(lctx, true);
        llama_batch_free(batch);
        if (decode_result != 0) {
            LOG_ERR("BitVLA language-model evaluation failed: %d\n", decode_result);
            return 1;
        }

        std::vector<float> hidden_states((size_t) n_action_tokens * n_embd);
        for (int i = 0; i < n_action_tokens; ++i) {
            const float * embedding = llama_get_embeddings_ith(lctx, action_start + i);
            if (!embedding) {
                LOG_ERR("missing hidden state for action token %d\n", i);
                return 1;
            }
            std::copy(embedding, embedding + n_embd, hidden_states.begin() + (size_t) i * n_embd);
        }
        if (!check_finite("action hidden states", hidden_states.data(), hidden_states.size())) {
            return 1;
        }

        std::vector<float> normalized_actions((size_t) info.action_chunk * info.action_dim);
        if (mtmd_predict_action(mtmd_ctx.get(), hidden_states.data(), normalized_actions.data()) != 0) {
            return 1;
        }
        if (!check_finite("normalized actions", normalized_actions.data(), normalized_actions.size())) {
            return 1;
        }

        json output;
        output["normalized_actions"] = action_matrix(normalized_actions, info.action_chunk, info.action_dim);
        if (use_stats) {
            const auto actions = unnormalize_actions(normalized_actions, stats, info.action_dim);
            output["actions"] = action_matrix(actions, info.action_chunk, info.action_dim);
        }
        output["prompt_tokens"] = n_tokens - n_image_tokens - n_action_tokens - 1;
        output["image_tokens"] = n_image_tokens;
        LOG_INF("BitVLA action prediction complete: %d prompt, %d image, %d action tokens\n",
            (int) output["prompt_tokens"], n_image_tokens, n_action_tokens);
        std::cout << std::setprecision(8) << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        LOG_ERR("%s\n", error.what());
        return 1;
    }
}
