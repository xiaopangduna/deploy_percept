#pragma once

#include "deploy_percept/data/coco_labels.hpp"
#include "deploy_percept/data/imagenet_labels.hpp"

#include <cstdint>

namespace deploy_percept {
namespace data {

enum class LabelVocabulary : std::uint8_t
{
    COCO_80,
    IMAGENET_1000,
};

using LabelLookupFn = const char *(*)(int class_id);

struct ClassLabelTable
{
    LabelVocabulary vocabulary;
    int num_classes;
    LabelLookupFn lookup;
};

inline const char *lookup_label(const ClassLabelTable &table, const int class_id)
{
    if (table.lookup == nullptr)
    {
        return nullptr;
    }
    return table.lookup(class_id);
}

inline constexpr ClassLabelTable kCoco80Labels{
    LabelVocabulary::COCO_80,
    kCocoNumClasses,
    coco_label,
};

inline constexpr ClassLabelTable kImageNet1000Labels{
    LabelVocabulary::IMAGENET_1000,
    kImageNetNumClasses,
    imagenet_label,
};

} // namespace data
} // namespace deploy_percept
