#pragma once

#include "ge_state.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <vector>

namespace mhp3rd::gpu {

// Experimental host-only silhouettes. No entity addresses or game assets are
// assumed. Weighted, opaque 3D draws are candidates, not verified actor IDs.
// Parts with identical world/view/projection matrices share a coverage mask.
struct PlanarShadowOptions {
    bool enabled{};
    bool gpu{};
    bool hide_original{true};
    bool trace{};
    float opacity{0.25f};
    float direction_x{0.45f};
    float direction_z{0.30f};
    float floor_offset{};
    unsigned resolution{96};

    static PlanarShadowOptions environment() {
        PlanarShadowOptions o;
        const auto flag = [](const char *name) {
            const char *s = std::getenv(name);
            return s && std::strcmp(s, "1") == 0;
        };
        const auto number = [](const char *name, float fallback, float low, float high) {
            const char *s = std::getenv(name);
            if (!s) return fallback;
            char *end{};
            const float value = std::strtof(s, &end);
            return end != s && *end == '\0' && std::isfinite(value) ? std::clamp(value, low, high) : fallback;
        };
        o.enabled = flag("MHP3RD_PLANAR_SHADOWS");
        o.gpu = flag("MHP3RD_SHADOW_GPU");
        o.hide_original = !flag("MHP3RD_KEEP_ORIGINAL_SHADOWS");
        o.trace = flag("MHP3RD_TRACE_SHADOWS");
        o.opacity = number("MHP3RD_SHADOW_OPACITY", o.opacity, 0.0f, 0.6f);
        o.direction_x = number("MHP3RD_SHADOW_X", o.direction_x, -2.0f, 2.0f);
        o.direction_z = number("MHP3RD_SHADOW_Z", o.direction_z, -2.0f, 2.0f);
        o.floor_offset = number("MHP3RD_SHADOW_FLOOR_OFFSET", 0.0f, -1000.0f, 1000.0f);
        o.resolution = static_cast<unsigned>(number("MHP3RD_SHADOW_RESOLUTION", 96.0f, 32.0f, 192.0f));
        return o;
    }
};

class PlanarShadows {
    using Point = std::array<float, 3>;
    using Triangle = std::array<Point, 3>;
    struct Caster {
        DrawCall state;
        std::array<float, 16> world;
        std::vector<Triangle> triangles;
    };
    std::vector<Caster> casters_;
    std::vector<Caster> receivers_;
    std::size_t receiver_triangles_{};
    std::size_t conformed_{};
    std::set<std::uint32_t> finished_;
    std::size_t triangles_{};
    std::size_t work_{};
    std::size_t emitted_{};
    std::size_t skipped_{};
    std::uint64_t frame_{};
    bool trace_water_{};
    unsigned trace_draw_{};
    static constexpr std::size_t kTriangleLimit = 40000;
    static constexpr std::size_t kPixelWorkLimit = 4000000;

    static bool same_view(const DrawCall &a, const DrawCall &b) {
        return a.target.color_address == b.target.color_address && a.view == b.view && a.projection == b.projection;
    }

    template<class Visit>
    static void visit_triangles(const DrawCall &call, Visit visit) {
        const auto point = [&](std::size_t index, Point &p) {
            if (!call.indices.empty()) index = call.indices[index];
            if (index >= call.vertices.size()) return false;
            const auto &v = call.vertices[index].position;
            for (unsigned k = 0; k < 3; ++k) {
                p[k] = call.world[k]*v[0] + call.world[4+k]*v[1] + call.world[8+k]*v[2] + call.world[12+k];
                if (!std::isfinite(p[k]) || std::abs(p[k]) > 1e7f) return false;
            }
            return true;
        };
        const std::size_t count = call.indices.empty() ? call.vertices.size() : call.indices.size();
        const std::size_t step = call.primitive == PrimitiveType::Triangles ? 3 : 1;
        for (std::size_t i = 0; i+2 < count; i += step) {
            Triangle t;
            const std::size_t a = call.primitive == PrimitiveType::TriangleFan ? 0 : i;
            if (point(a,t[0]) && point(i+1,t[1]) && point(i+2,t[2]) && !visit(t)) break;
        }
    }

    static float edge(const Point &a, const Point &b, float x, float z) {
        return (b[0]-a[0])*(z-a[2]) - (b[2]-a[2])*(x-a[0]);
    }

    static bool height_at(const Triangle &t, float x, float z, float &height) {
        const float area = edge(t[0],t[1],t[2][0],t[2][2]);
        if (std::abs(area) < 1e-8f) return false;
        const float a = edge(t[1],t[2],x,z)/area;
        const float b = edge(t[2],t[0],x,z)/area;
        const float c = 1-a-b;
        if (a < -1e-5f || b < -1e-5f || c < -1e-5f) return false;
        height = a*t[0][1]+b*t[1][1]+c*t[2][1];
        return std::isfinite(height);
    }

    void collect_receiver(const DrawCall &call) {
        // Depth-writing unweighted surfaces are receiver candidates even with
        // blending enabled. The captured scene leaves blending on for terrain.
        // Alpha testing may be globally enabled on opaque terrain; do not
        // reject it just for that flag. Cutout textures remain approximate.
        auto it = std::find_if(receivers_.begin(),receivers_.end(),[&](const Caster &r) {return same_view(r.state,call);});
        if (it == receivers_.end()) {
            if (receivers_.size() >= 16) return;
            Caster r;
            r.state.target=call.target; r.state.view=call.view; r.state.projection=call.projection;
            receivers_.push_back(std::move(r)); it=receivers_.end()-1;
        }
        visit_triangles(call,[&](const Triangle &t) {
            if (receiver_triangles_ >= 60000) return false;
            Point a{},b{},normal{};
            for (unsigned k=0;k<3;++k) {a[k]=t[1][k]-t[0][k];b[k]=t[2][k]-t[0][k];}
            normal={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
            const float length=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
            if (std::isfinite(length) && length>1e-8f && std::abs(normal[1])>0.35f*length) {
                it->triangles.push_back(t); ++receiver_triangles_;
            }
            return true;
        });
    }

    // Sutherland-Hodgman clipping in world X/Z. Y interpolates along the
    // receiver triangle, so the emitted polygon lies on the actual surface.
    struct ClippedPolygon {
        std::array<Point, 8> points{}; // Triangle clipped by four half-planes: at most seven vertices.
        std::size_t count{};
        bool empty() const { return count==0; }
        std::size_t size() const { return count; }
        const Point &back() const { return points[count-1]; }
        const Point *begin() const { return points.data(); }
        const Point *end() const { return points.data()+count; }
        const Point &operator[](std::size_t i) const { return points[i]; }
        void push_back(const Point &p) { points[count++]=p; }
    };
    static ClippedPolygon clip_rectangle(const Triangle &t,float x0,float z0,float x1,float z1) {
        ClippedPolygon polygon;
        for(const auto &p:t) polygon.push_back(p);
        const auto cut=[&](unsigned axis,float bound,bool greater) {
            ClippedPolygon next;
            if (polygon.empty()) return;
            Point a=polygon.back();
            float da=greater ? a[axis]-bound : bound-a[axis];
            for (const Point &b:polygon) {
                const float db=greater ? b[axis]-bound : bound-b[axis];
                if ((da>=0)!=(db>=0)) {
                    const float ratio=da/(da-db);
                    Point hit{};
                    for(unsigned k=0;k<3;++k) hit[k]=a[k]+ratio*(b[k]-a[k]);
                    hit[axis]=bound;
                    next.push_back(hit);
                }
                if(db>=0) next.push_back(b);
                a=b;da=db;
            }
            polygon=std::move(next);
        };
        cut(0,x0,true);cut(0,x1,false);cut(2,z0,true);cut(2,z1,false);
        return polygon;
    }

    DrawCall rasterize(const Caster &caster) {
        DrawCall out = caster.state;
        const unsigned n = options.resolution;
        float floor = std::numeric_limits<float>::max();
        float top = -floor;
        for (const auto &t : caster.triangles) for (const auto &p : t) {
            floor = std::min(floor, p[1]);
            top = std::max(top, p[1]);
        }
        if (!(top > floor + 0.001f)) return out;
        const float height = top-floor;
        // Ground under the actor stabilizes the projection reference instead
        // of following a moving toe every frame. No collision data is read.
        float centre_x=0,centre_z=0;std::size_t points=0;
        for(const auto &t:caster.triangles) for(const auto &p:t) {centre_x+=p[0];centre_z+=p[2];++points;}
        centre_x/=static_cast<float>(points);centre_z/=static_cast<float>(points);
        float ground=-std::numeric_limits<float>::infinity();
        for(const Caster &r:receivers_) if(same_view(r.state,caster.state)) for(const auto &t:r.triangles) {
            float y{};
            if(height_at(t,centre_x,centre_z,y) && y<=floor+height*0.35f && y>=floor-height*2 && y>ground) ground=y;
        }
        if(std::isfinite(ground)) floor=ground;
        floor += options.floor_offset;
        const float bias = std::max(0.02f, height * 0.002f);
        std::vector<Triangle> projected = caster.triangles;
        float x0 = std::numeric_limits<float>::max(), z0 = x0;
        float x1 = -x0, z1 = -x0;
        for (auto &t : projected) for (auto &p : t) {
            const float h = p[1] - floor;
            p = {p[0] + options.direction_x * h, floor + bias, p[2] + options.direction_z * h};
            x0 = std::min(x0, p[0]); x1 = std::max(x1, p[0]);
            z0 = std::min(z0, p[2]); z1 = std::max(z1, p[2]);
        }
        if (!(x1 > x0 + 0.001f && z1 > z0 + 0.001f)) return out;
        // Square world-aligned cells: small pose changes no longer resize and
        // stretch every texel as an animated bounding box changes. Quantized
        // scale can still step when an actor crosses a power-of-two extent.
        const float span=std::max(x1-x0,z1-z0)/static_cast<float>(n-8);
        const float dx=std::exp2(std::ceil(std::log2(std::max(span,1.0f/1024.0f))));
        const float dz=dx;
        x0=(std::floor(x0/dx)-2)*dx;z0=(std::floor(z0/dz)-2)*dz;
        if(options.gpu) {
            if(mask_jobs.size()>=48) {++skipped_;return out;}
            MaskJob job;job.resolution=n;job.points.reserve(projected.size()*3);
            for(const auto &t:projected) for(const auto &p:t)
                job.points.push_back({(p[0]-x0)/dx,(p[2]-z0)/dz,0,0});
            // Submit nearby receiver triangles once. UV bounds and mask coverage
            // are evaluated per fragment on the GPU; no per-cell CPU clipping.
            const auto alpha=static_cast<std::uint32_t>(std::lround(options.opacity*255));
            for(const Caster &r:receivers_) if(same_view(r.state,caster.state)) for(const auto &t:r.triangles) {
                if(std::min({t[0][1],t[1][1],t[2][1]})>floor+height*.5f ||
                   std::max({t[0][1],t[1][1],t[2][1]})<floor-height*2 ||
                   std::max({t[0][0],t[1][0],t[2][0]})<x0 ||
                   std::min({t[0][0],t[1][0],t[2][0]})>x0+n*dx ||
                   std::max({t[0][2],t[1][2],t[2][2]})<z0 ||
                   std::min({t[0][2],t[1][2],t[2][2]})>z0+n*dz) continue;
                if(out.vertices.size()+3>120000) {++skipped_;break;}
                for(const auto &p:t) {
                    Vertex v;v.position={p[0],p[1]+bias+options.floor_offset,p[2],1};
                    v.texcoord={(p[0]-x0)/(n*dx),(p[2]-z0)/(n*dz)};
                    v.color=alpha<<24u;out.vertices.push_back(v);
                }
                ++conformed_;
            }
            out.texture.enabled=true;out.texture.function=6;
            out.texture.address=0xF1000000u+static_cast<unsigned>(mask_jobs.size());
            out.texture.width=out.texture.height=static_cast<std::uint16_t>(n);
            out.texture.scale_u=out.texture.scale_v=1;
            mask_jobs.push_back(std::move(job));
            return out;
        }
        // Bin nearby receivers by mask row to bound clipping work.
        std::vector<std::vector<const Triangle *>> rows(n);
        for(const Caster &r:receivers_) if(same_view(r.state,caster.state)) for(const auto &t:r.triangles) {
            const float low_y=std::min({t[0][1],t[1][1],t[2][1]});
            const float high_y=std::max({t[0][1],t[1][1],t[2][1]});
            const float left=std::min({t[0][0],t[1][0],t[2][0]});
            const float right=std::max({t[0][0],t[1][0],t[2][0]});
            const float low=std::min({t[0][2],t[1][2],t[2][2]});
            const float high=std::max({t[0][2],t[1][2],t[2][2]});
            if(low_y>floor+height*0.5f || high_y<floor-height*2 || right<x0 || left>x0+n*dx || high<z0 || low>z0+n*dz) continue;
            const int first=int(std::clamp(std::floor((low-z0)/dz),0.0f,float(n-1)));
            const int last=int(std::clamp(std::floor((high-z0)/dz),0.0f,float(n-1)));
            for(int row=first;row<=last;++row) rows[row].push_back(&t);
        }
        std::vector<unsigned char> mask(n * n, 0);
        for (auto &t : projected) {
            for (auto &p : t) { p[0] = (p[0] - x0) / dx; p[2] = (p[2] - z0) / dz; }
            if (std::abs(edge(t[0], t[1], t[2][0], t[2][2])) < 1e-7f) continue;
            const int left = std::clamp(static_cast<int>(std::floor(std::min({t[0][0], t[1][0], t[2][0]}))), 0, int(n)-1);
            const int right = std::clamp(static_cast<int>(std::ceil(std::max({t[0][0], t[1][0], t[2][0]}))), 0, int(n)-1);
            const int low = std::clamp(static_cast<int>(std::floor(std::min({t[0][2], t[1][2], t[2][2]}))), 0, int(n)-1);
            const int high = std::clamp(static_cast<int>(std::ceil(std::max({t[0][2], t[1][2], t[2][2]}))), 0, int(n)-1);
            work_ += static_cast<std::size_t>(right-left+1) * static_cast<std::size_t>(high-low+1);
            if (work_ > kPixelWorkLimit) { ++skipped_; return out; }
            for (int z = low; z <= high; ++z) for (int x = left; x <= right; ++x) {
                // Union is monotonic: samples covered by an earlier triangle
                // cannot change. Avoid repeating their edge tests.
                auto &cell=mask[z*n+x];
                if(cell==15u) continue;
                // Four coverage bits union across triangles before filtering.
                // This gives partial coverage without a hard one-sample toggle.
                for(unsigned sample=0;sample<4;++sample) {
                    if(cell & (1u<<sample)) continue;
                    const float sx=x+((sample&1)?0.75f:0.25f);
                    const float sz=z+((sample&2)?0.75f:0.25f);
                    const float a=edge(t[0],t[1],sx,sz),b=edge(t[1],t[2],sx,sz),c=edge(t[2],t[0],sx,sz);
                    if((a>=0 && b>=0 && c>=0)||(a<=0 && b<=0 && c<=0)) mask[z*n+x]|=1u<<sample;
                }
            }
        }
        std::vector<unsigned char> coverage(n * n, 0);
        for (unsigned z = 1; z+1 < n; ++z) for (unsigned x = 1; x+1 < n; ++x) {
            unsigned sum = 0;
            for (int j = -1; j <= 1; ++j) for (int i = -1; i <= 1; ++i)
                sum += std::popcount(static_cast<unsigned>(mask[(int(z)+j)*n+int(x)+i]));
            coverage[z*n+x] = static_cast<unsigned char>(sum);
        }
        // Clip each constant-alpha mask run against actual receiver triangles.
        // Do not bridge holes or use a floating fallback when terrain is absent.
        for (unsigned z=0;z<n;++z) for(unsigned x=0;x<n;) {
            const unsigned strength=coverage[z*n+x];
            if(!strength) {++x;continue;}
            const unsigned start=x++;
            while(x<n && coverage[z*n+x]==strength) ++x;
            const auto alpha=static_cast<std::uint32_t>(std::lround(options.opacity*255.0f*strength/36.0f));
            if(!alpha) continue;
            const float left=x0+start*dx,right=x0+x*dx,low=z0+z*dz,high=low+dz;
            for(const Triangle *t:rows[z]) {
                if(++work_>kPixelWorkLimit) {++skipped_;return out;}
                if(std::max({(*t)[0][0],(*t)[1][0],(*t)[2][0]})<left ||
                   std::min({(*t)[0][0],(*t)[1][0],(*t)[2][0]})>right) continue;
                auto polygon=clip_rectangle(*t,left,low,right,high);
                if(polygon.size()<3) continue;
                const auto vertex=[&](const Point &p) {
                    Vertex v;v.position={p[0],p[1]+bias+options.floor_offset,p[2],1};v.color=alpha<<24u;return v;
                };
                for(std::size_t k=1;k+1<polygon.size();++k) {
                    if(std::abs(edge(polygon[0],polygon[k],polygon[k+1][0],polygon[k+1][2]))<1e-9f) continue;
                    // Bound extra vertex memory independently from input size.
                    if(out.vertices.size()+3>120000) {++skipped_;return out;}
                    out.vertices.insert(out.vertices.end(),{vertex(polygon[0]),vertex(polygon[k]),vertex(polygon[k+1])});
                    ++conformed_;
                }
            }
        }
        return out;
    }

public:
    struct MaskJob { unsigned resolution; std::vector<std::array<float,4>> points; };
    std::vector<MaskJob> mask_jobs;
    PlanarShadowOptions options;
    bool drawing{};
    PlanarShadows() : PlanarShadows(PlanarShadowOptions::environment()) {}
    explicit PlanarShadows(PlanarShadowOptions o) : options(o) {}

    void begin_frame(std::uint64_t frame) {
        mask_jobs.clear();
        // A menu-only frame must not consume the requested scene capture.
        const bool waiting_for_scene=trace_water_ && trace_draw_==0;
        if (trace_water_ && !waiting_for_scene) std::cout << "[shadow-water] end\n" << std::flush;
        trace_water_=waiting_for_scene; trace_draw_=0;
        // Explicit one-frame capture, requested from the working directory.
        if (options.enabled && std::getenv("MHP3RD_TRACE_SHADOW_WATER")) {
            if (FILE *trigger=std::fopen("shadow-trace.trigger","rb")) {
                std::fclose(trigger);
                if (std::remove("shadow-trace.trigger")==0) {
                    trace_water_=true;
                    std::cout << "[shadow-water] begin frame=" << frame << '\n';
                }
            }
        }
        if (options.enabled && options.trace && frame % 120u == 0u)
            std::cout << "[shadows] previous frame: triangles=" << triangles_ << " silhouettes=" << emitted_
                      << " receiver triangles=" << receiver_triangles_ << " conformed=" << conformed_
                      << " skipped=" << skipped_ << " raster samples=" << work_ << '\n';
        casters_.clear(); receivers_.clear(); finished_.clear();
        receiver_triangles_=conformed_=0;triangles_ = work_ = emitted_ = skipped_ = 0; frame_ = frame;
    }

    void clear_target(std::uint32_t target) {
        std::erase_if(casters_, [&](const Caster &c) { return c.state.target.color_address == target; });
        std::erase_if(receivers_, [&](const Caster &c) { return c.state.target.color_address == target; });
        finished_.erase(target);
    }

    // Conservative signature from the circle capture. No guest addresses:
    // texture allocations change between maps. This is a heuristic, not an
    // actor ID; the keep-original switch permits comparison and recovery.
    bool hides_original(const DrawCall &c) const {
        if(!options.enabled || !options.hide_original || options.opacity<=0 || drawing ||
           c.clear_mode || c.through || c.raw_vertices || c.lighting_enabled ||
           c.has_vertex_color || ((c.vertex_type>>9u)&3u)!=0u ||
           !c.depth.test_enabled || c.depth.write_enabled || c.depth.function!=5u ||
           !c.blend.enabled || c.blend.equation!=0u || c.blend.source_factor!=2u ||
           c.blend.destination_factor!=3u || !c.texture.enabled ||
           c.texture.width!=64 || c.texture.height!=64 || c.texture.format!=TextureFormat::Clut8 ||
           c.texture.function!=0u || !c.texture.alpha_from_texture || !c.alpha_test.enabled ||
           (c.material_color&0xFFFFFFu)!=0u || (c.material_color>>24u)==0u ||
           (c.primitive!=PrimitiveType::TriangleStrip && c.primitive!=PrimitiveType::Triangles) ||
           c.vertices.size()<3 || c.vertices.size()>256) return false;
        Point lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f};
        for(const auto &v:c.vertices) for(unsigned k=0;k<3;++k) {
            const float p=c.world[k]*v.position[0]+c.world[4+k]*v.position[1]+c.world[8+k]*v.position[2]+c.world[12+k];
            if(!std::isfinite(p)) return false;
            lo[k]=std::min(lo[k],p);hi[k]=std::max(hi[k],p);
        }
        return hi[1]-lo[1]<=0.05f && hi[0]-lo[0]>0.01f && hi[2]-lo[2]>0.01f;
    }

    void collect(const DrawCall &call) {
        if (trace_water_ && !drawing && trace_draw_++<12000) {
            std::cout << "[shadow-water] draw=" << trace_draw_
                      << " target=" << call.target.color_address
                      << " texture=" << call.texture.address
                      << " weighted=" << ((call.vertex_type>>9u)&3u)
                      << " through=" << call.through << " clear=" << call.clear_mode
                      << " blend=" << call.blend.enabled
                      << " equation=" << call.blend.equation
                      << " factors=" << call.blend.source_factor << ',' << call.blend.destination_factor
                      << " fixed=" << call.blend.fixed_source << ',' << call.blend.fixed_destination
                      << " lit=" << call.lighting_enabled
                      << " vertex_color=" << call.has_vertex_color
                      << " material=" << call.material_color
                      << " color0=" << (call.vertices.empty()?0u:call.vertices.front().color)
                      << " texsize=" << call.texture.width << 'x' << call.texture.height
                      << " texformat=" << static_cast<unsigned>(call.texture.format)
                      << " texfunc=" << call.texture.function
                      << " texalpha=" << call.texture.alpha_from_texture
                      << " ztest=" << call.depth.test_enabled << " zwrite=" << call.depth.write_enabled
                      << " zfunc=" << call.depth.function << " alpha=" << call.alpha_test.enabled
                      << " finished=" << finished_.contains(call.target.color_address)
                      << " vertices=" << call.vertices.size() << " primitive=" << int(call.primitive);
            Point low{1e30f,1e30f,1e30f},high{-1e30f,-1e30f,-1e30f};
            if (!call.through && !call.raw_vertices &&
                (call.primitive==PrimitiveType::Triangles || call.primitive==PrimitiveType::TriangleStrip ||
                 call.primitive==PrimitiveType::TriangleFan)) {
                visit_triangles(call,[&](const Triangle &t) {
                    for(const auto &p:t) for(unsigned k=0;k<3;++k) {
                        low[k]=std::min(low[k],p[k]);high[k]=std::max(high[k],p[k]);
                    }
                    return true;
                });
            }
            std::cout << " bounds=" << low[0] << ',' << low[1] << ',' << low[2]
                      << ':' << high[0] << ',' << high[1] << ',' << high[2] << '\n';
        }
        if (!options.enabled || options.opacity == 0.0f || drawing || call.clear_mode || call.through ||
            call.raw_vertices || call.vertices.empty() || !call.depth.test_enabled ||
            !call.depth.write_enabled ||
            finished_.contains(call.target.color_address)) return;
        if (call.primitive != PrimitiveType::Triangles && call.primitive != PrimitiveType::TriangleStrip &&
            call.primitive != PrimitiveType::TriangleFan) return;
        for (float value : call.world) if (!std::isfinite(value)) return;
        for (float value : call.view) if (!std::isfinite(value)) return;
        for (float value : call.projection) if (!std::isfinite(value)) return;
        if(((call.vertex_type>>9u)&3u)==0u) {collect_receiver(call);return;}
        if(call.blend.enabled) return; // Keep the existing opaque caster filter.
        auto it = std::find_if(casters_.begin(), casters_.end(), [&](const Caster &c) {
            return c.world == call.world && same_view(c.state, call);
        });
        if (it == casters_.end()) {
            if (casters_.size() >= 48) { ++skipped_; return; }
            Caster c;
            c.world = call.world;
            c.state.target = call.target; c.state.viewport = call.viewport;
            c.state.view = call.view; c.state.projection = call.projection;
            c.state.world = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            c.state.depth = call.depth; c.state.depth.write_enabled = false;
            c.state.blend = {true, 2u, 3u, 0u, 0u, 0u};
            c.state.has_vertex_color = true; c.state.primitive = PrimitiveType::Triangles;
            // Synthetic addresses separate these draws from game draws during
            // replay; no guest memory is read through these addresses.
            c.state.vertex_address = 0xF0000000u + static_cast<std::uint32_t>(casters_.size()) * 0x10000u;
            casters_.push_back(std::move(c)); it = casters_.end()-1;
        }
        visit_triangles(call,[&](const Triangle &t) {
            if(triangles_>=kTriangleLimit) {++skipped_;return false;}
            it->triangles.push_back(t);++triangles_;return true;
        });
    }

    std::vector<DrawCall> take(std::uint32_t target) {
        std::vector<DrawCall> draws;
        if (trace_water_) std::cout << "[shadow-water] flush target=" << target
            << " after=" << trace_draw_ << " casters=" << casters_.size()
            << " receivers=" << receiver_triangles_ << '\n';
        // 2D scenery may precede the first actor. Keep receiver geometry until
        // a scene with actual candidates is flushed or the target is cleared.
        if(std::none_of(casters_.begin(),casters_.end(),[&](const Caster &c) {
            return c.state.target.color_address==target;
        })) return draws;
        for (const Caster &caster : casters_) {
            if (caster.state.target.color_address != target || caster.triangles.empty()) continue;
            auto draw = rasterize(caster);
            if (!draw.vertices.empty()) { ++emitted_; draws.push_back(std::move(draw)); }
        }
        if (!draws.empty() && options.trace && frame_ % 120u == 0u)
            std::cout << "[shadows] target=" << target << " silhouettes=" << draws.size()
                      << " (weighted-draw heuristic, clipped to visible opaque terrain)\n";
        const bool had = std::any_of(casters_.begin(), casters_.end(), [&](const Caster &c) {
            return c.state.target.color_address == target;
        });
        clear_target(target);
        if (had) finished_.insert(target);
        return draws;
    }

    std::vector<std::uint32_t> pending_targets() const {
        std::set<std::uint32_t> targets;
        for (const auto &c : casters_) targets.insert(c.state.target.color_address);
        return {targets.begin(), targets.end()};
    }
};

} // namespace mhp3rd::gpu
