#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace mhp3rd::gpu {
// Resources follow the renderer's fenced vertex regions, including replay.
// No readback or queue-idle wait is used in the rendering path.
class ShadowGpu {
    struct Item {
        VkBuffer buffer{}; VkDeviceMemory buffer_memory{}; void *mapped{};
        VkDeviceSize capacity{};
        VkImage image{}; VkDeviceMemory image_memory{}; VkImageView view{};
        VkDescriptorSet compute{},sample{};
        unsigned resolution{}; bool initialized{};
    };
    VkDevice device_{}; VkPhysicalDevice physical_{};
    VkDescriptorSetLayout layout_{}; VkPipelineLayout pipeline_layout_{};
    VkPipeline pipeline_{}; VkDescriptorPool pool_{};
    VkDescriptorSetLayout sample_layout_{}; VkSampler sampler_{};
    std::array<std::vector<Item>,3> regions_;
    unsigned region_{},cursor_{};
    static void check(VkResult r) {if(r!=VK_SUCCESS) throw std::runtime_error("GPU shadow Vulkan resource creation failed");}
    unsigned memory_type(unsigned bits,VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties m{};vkGetPhysicalDeviceMemoryProperties(physical_,&m);
        for(unsigned i=0;i<m.memoryTypeCount;++i) if((bits&(1u<<i)) && (m.memoryTypes[i].propertyFlags&flags)==flags) return i;
        throw std::runtime_error("GPU shadow memory type unavailable");
    }
    void allocate(VkMemoryRequirements req,VkMemoryPropertyFlags flags,VkDeviceMemory &memory) {
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};a.allocationSize=req.size;
        a.memoryTypeIndex=memory_type(req.memoryTypeBits,flags);check(vkAllocateMemory(device_,&a,nullptr,&memory));
    }
    void release(Item &i) {
        if(i.mapped) vkUnmapMemory(device_,i.buffer_memory);
        vkDestroyBuffer(device_,i.buffer,nullptr);vkFreeMemory(device_,i.buffer_memory,nullptr);
        vkDestroyImageView(device_,i.view,nullptr);vkDestroyImage(device_,i.image,nullptr);vkFreeMemory(device_,i.image_memory,nullptr);
        if(i.compute) vkFreeDescriptorSets(device_,pool_,1,&i.compute);
        if(i.sample) vkFreeDescriptorSets(device_,pool_,1,&i.sample);
        i={};
    }
public:
    void begin(unsigned region) {region_=region;cursor_=0;}
    void shutdown() {
        if(!device_) return;
        for(auto &r:regions_) {for(auto &i:r) release(i);r.clear();}
        vkDestroyPipeline(device_,pipeline_,nullptr);vkDestroyPipelineLayout(device_,pipeline_layout_,nullptr);
        vkDestroyDescriptorPool(device_,pool_,nullptr);vkDestroyDescriptorSetLayout(device_,layout_,nullptr);
        pipeline_={};pipeline_layout_={};pool_={};layout_={};device_={};
    }
    void initialize(VkDevice device,VkPhysicalDevice physical,unsigned queue_family,
                    VkDescriptorSetLayout sample_layout,VkSampler sampler,const std::uint32_t *code,std::size_t bytes) {
        if(device_) return;
        unsigned count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);
        std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        if(queue_family>=count || !(families[queue_family].queueFlags&VK_QUEUE_COMPUTE_BIT))
            throw std::runtime_error("GPU shadows require compute on the graphics queue");
        device_=device;physical_=physical;sample_layout_=sample_layout;sampler_=sampler;
        try {
            std::array<VkDescriptorSetLayoutBinding,2> b{{{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},
                {1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}}};
            VkDescriptorSetLayoutCreateInfo l{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};l.bindingCount=2;l.pBindings=b.data();
            check(vkCreateDescriptorSetLayout(device_,&l,nullptr,&layout_));
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,8};
            VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&layout_;
            pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&range;check(vkCreatePipelineLayout(device_,&pl,nullptr,&pipeline_layout_));
            std::array<VkDescriptorPoolSize,3> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,144},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,144},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,144}}};
            VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            pool.maxSets=288;pool.poolSizeCount=3;pool.pPoolSizes=sizes.data();check(vkCreateDescriptorPool(device_,&pool,nullptr,&pool_));
            VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=bytes;sm.pCode=code;
            VkShaderModule module{};check(vkCreateShaderModule(device_,&sm,nullptr,&module));
            VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.layout=pipeline_layout_;
            cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=module;cp.stage.pName="main";
            VkResult result=vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&cp,nullptr,&pipeline_);vkDestroyShaderModule(device_,module,nullptr);check(result);
            std::cout<<"[shadows-gpu] compute masks and filtered receivers enabled; CPU geometry collection retained\n";
        } catch(...) {shutdown();throw;}
    }
    VkDescriptorSet render(VkCommandBuffer command,unsigned n,const std::vector<std::array<float,4>> &points) {
        if(!device_ || region_>=3 || cursor_>=48 || points.empty()) return VK_NULL_HANDLE;
        auto &r=regions_[region_];if(cursor_==r.size()) r.emplace_back();Item &i=r[cursor_++];
        const VkDeviceSize bytes=points.size()*sizeof(points[0]);
        if(i.resolution!=n || i.capacity<bytes) {
            release(i);
            try {
                i.capacity=std::max<VkDeviceSize>(bytes,65536);i.resolution=n;
                VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=i.capacity;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                check(vkCreateBuffer(device_,&bi,nullptr,&i.buffer));VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device_,i.buffer,&req);
                allocate(req,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,i.buffer_memory);
                check(vkBindBufferMemory(device_,i.buffer,i.buffer_memory,0));check(vkMapMemory(device_,i.buffer_memory,0,VK_WHOLE_SIZE,0,&i.mapped));
                VkImageCreateInfo im{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};im.imageType=VK_IMAGE_TYPE_2D;im.format=VK_FORMAT_R8G8B8A8_UNORM;
                im.extent={n,n,1};im.mipLevels=im.arrayLayers=1;im.samples=VK_SAMPLE_COUNT_1_BIT;im.tiling=VK_IMAGE_TILING_OPTIMAL;
                im.usage=VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;check(vkCreateImage(device_,&im,nullptr,&i.image));
                vkGetImageMemoryRequirements(device_,i.image,&req);allocate(req,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,i.image_memory);
                check(vkBindImageMemory(device_,i.image,i.image_memory,0));
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=i.image;vi.viewType=VK_IMAGE_VIEW_TYPE_2D;vi.format=im.format;
                vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};check(vkCreateImageView(device_,&vi,nullptr,&i.view));
                VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=pool_;da.descriptorSetCount=1;da.pSetLayouts=&layout_;
                check(vkAllocateDescriptorSets(device_,&da,&i.compute));da.pSetLayouts=&sample_layout_;check(vkAllocateDescriptorSets(device_,&da,&i.sample));
                VkDescriptorBufferInfo db{i.buffer,0,i.capacity};
                VkDescriptorImageInfo storage{VK_NULL_HANDLE,i.view,VK_IMAGE_LAYOUT_GENERAL},sample{sampler_,i.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                std::array<VkWriteDescriptorSet,3> w{};
                for(auto &v:w){v.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;v.descriptorCount=1;}
                w[0].dstSet=i.compute;w[0].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w[0].pBufferInfo=&db;
                w[1].dstSet=i.compute;w[1].dstBinding=1;w[1].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;w[1].pImageInfo=&storage;
                w[2].dstSet=i.sample;w[2].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w[2].pImageInfo=&sample;
                vkUpdateDescriptorSets(device_,3,w.data(),0,nullptr);
            } catch(...) {release(i);throw;}
        }
        std::memcpy(i.mapped,points.data(),static_cast<std::size_t>(bytes));
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};barrier.image=i.image;
        barrier.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.oldLayout=i.initialized?VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout=VK_IMAGE_LAYOUT_GENERAL;barrier.srcAccessMask=i.initialized?VK_ACCESS_SHADER_READ_BIT:0;barrier.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command,i.initialized?VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT:VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_);
        vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout_,0,1,&i.compute,0,nullptr);
        const std::array<unsigned,2> params{static_cast<unsigned>(points.size()/3),n};
        vkCmdPushConstants(command,pipeline_layout_,VK_SHADER_STAGE_COMPUTE_BIT,0,8,params.data());
        vkCmdDispatch(command,(n+7)/8,(n+7)/8,1);
        barrier.oldLayout=VK_IMAGE_LAYOUT_GENERAL;barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
        i.initialized=true;return i.sample;
    }
};
}
