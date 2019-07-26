/*
For more information, please see: http://software.sci.utah.edu

The MIT License

Copyright (c) 2015 Scientific Computing and Imaging Institute,
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#include <Graphics/Glyphs/GlyphGeom.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Math/MiscMath.h>
#include <Core/GeometryPrimitives/Transform.h>

using namespace SCIRun;
using namespace Graphics;
using namespace Datatypes;
using namespace Core::Geometry;
using namespace Core::Datatypes;

GlyphGeom::GlyphGeom() : numVBOElements_(0), lineIndex_(0)
{

}

void GlyphGeom::buildObject(GeometryObjectSpire& geom, const std::string& uniqueNodeID, const bool isTransparent, const double transparencyValue,
  const ColorScheme& colorScheme, RenderState state, const SpireIBO::PRIMITIVE& primIn, const BBox& bbox)
{
  std::string vboName = uniqueNodeID + "VBO";
  std::string iboName = uniqueNodeID + "IBO";
  std::string passName = uniqueNodeID + "Pass";

  bool useTriangles = primIn == SpireIBO::PRIMITIVE::TRIANGLES;

  // Construct VBO.
  std::string shader = "Shaders/UniformColor";
  std::vector<SpireVBO::AttributeData> attribs;
  attribs.push_back(SpireVBO::AttributeData("aPos", 3 * sizeof(float)));
  if (useTriangles)
    attribs.push_back(SpireVBO::AttributeData("aNormal", 3 * sizeof(float)));
  RenderType renderType = RenderType::RENDER_VBO_IBO;

  //ColorScheme colorScheme = COLOR_UNIFORM;

  std::vector<SpireSubPass::Uniform> uniforms;
  if (isTransparent)
    uniforms.push_back(SpireSubPass::Uniform("uTransparency", static_cast<float>(transparencyValue)));
  // TODO: add colormapping options
  if (colorScheme == ColorScheme::COLOR_MAP)
  {
    attribs.push_back(SpireVBO::AttributeData("aColor", 4 * sizeof(float)));
    if (useTriangles)
    {
      shader = "Shaders/DirPhongCMap";
      uniforms.push_back(SpireSubPass::Uniform("uAmbientColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f)));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 0.1f)));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularPower", 32.0f));
    }
    else
    {
      shader = "Shaders/ColorMap";
    }
  }
  else if (colorScheme == ColorScheme::COLOR_IN_SITU)
  {
    attribs.push_back(SpireVBO::AttributeData("aColor", 4 * sizeof(float)));
    if (useTriangles)
    {
      shader = "Shaders/DirPhongInSitu";
      uniforms.push_back(SpireSubPass::Uniform("uAmbientColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f)));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 0.1f)));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularPower", 32.0f));
    }
    else
    {
      shader = "Shaders/InSituColor";
    }
  }
  else if (colorScheme == ColorScheme::COLOR_UNIFORM)
  {
    ColorRGB dft = state.defaultColor;
    if (useTriangles)
    {
      if (geom.isClippable())
        shader = "Shaders/DirPhong";
      else
        shader = "Shaders/DirPhongNoClipping";
      uniforms.push_back(SpireSubPass::Uniform("uAmbientColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 1.0f)));
      uniforms.push_back(SpireSubPass::Uniform("uDiffuseColor",
        glm::vec4(dft.r(), dft.g(), dft.b(), static_cast<float>(transparencyValue))));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularColor",
        glm::vec4(0.1f, 0.1f, 0.1f, 0.1f)));
      uniforms.push_back(SpireSubPass::Uniform("uSpecularPower", 32.0f));
    }
    else
    {
      uniforms.emplace_back("uColor", glm::vec4(dft.r(), dft.g(), dft.b(), static_cast<float>(transparencyValue)));
    }
  }

  uint32_t iboSize = 0;
  uint32_t vboSize = 0;

  vboSize = static_cast<uint32_t>(points_.size()) * 3 * sizeof(float);
  vboSize += static_cast<uint32_t>(normals_.size()) * 3 * sizeof(float);
  if (colorScheme == ColorScheme::COLOR_IN_SITU || colorScheme == ColorScheme::COLOR_MAP)
    vboSize += static_cast<uint32_t>(colors_.size()) * 4 * sizeof(float); //RGBA
  iboSize = static_cast<uint32_t>(indices_.size()) * sizeof(uint32_t);
  /// \todo To reduce memory requirements, we can use a 16bit index buffer.

  /// \todo To further reduce a large amount of memory, get rid of the index
  ///       buffer and use glDrawArrays to render without an IBO. An IBO is
  ///       a waste of space.
  ///       http://www.opengl.org/sdk/docs/man3/xhtml/glDrawArrays.xml

  /// \todo Switch to unique_ptrs and move semantics.
  std::shared_ptr<spire::VarBuffer> iboBufferSPtr(new spire::VarBuffer(iboSize));
  std::shared_ptr<spire::VarBuffer> vboBufferSPtr(new spire::VarBuffer(vboSize));

  // Accessing the pointers like this is contrived. We only do this for
  // speed since we will be using the pointers in a tight inner loop.
  auto iboBuffer = iboBufferSPtr.get();
  auto vboBuffer = vboBufferSPtr.get();

  //write to the IBO/VBOs

  for (auto a : indices_)
    iboBuffer->write(a);

  BBox newBBox;

  const bool writeNormals = normals_.size() == points_.size();
  for (size_t i = 0; i < points_.size(); i++)
  {
    // Write first point on line
    vboBuffer->write(static_cast<float>(points_.at(i).x()));
    vboBuffer->write(static_cast<float>(points_.at(i).y()));
    vboBuffer->write(static_cast<float>(points_.at(i).z()));

    newBBox.extend(Point(points_.at(i).x(), points_.at(i).y(), points_.at(i).z()));

    if (writeNormals)
    {
      vboBuffer->write(static_cast<float>(normals_.at(i).x()));
      vboBuffer->write(static_cast<float>(normals_.at(i).y()));
      vboBuffer->write(static_cast<float>(normals_.at(i).z()));
    }
    if (colorScheme == ColorScheme::COLOR_MAP || colorScheme == ColorScheme::COLOR_IN_SITU)
    {
      vboBuffer->write(static_cast<float>(colors_.at(i).r()));
      vboBuffer->write(static_cast<float>(colors_.at(i).g()));
      vboBuffer->write(static_cast<float>(colors_.at(i).b()));
      vboBuffer->write(static_cast<float>(colors_.at(i).a()));
      //vboBuffer->write(static_cast<float>(1.f));
    } // no color writing otherwise
  }

  if(!bbox.valid())
    newBBox.reset();

  // If true, then the VBO will be placed on the GPU. We don't want to place
  // VBOs on the GPU when we are generating rendering lists.
  SpireVBO geomVBO(vboName, attribs, vboBufferSPtr, numVBOElements_, newBBox, true);

  // Construct IBO.
  SpireIBO geomIBO(iboName, primIn, sizeof(uint32_t), iboBufferSPtr);

  state.set(RenderState::IS_ON, true);
  state.set(RenderState::HAS_DATA, true);

  SpireText text;

  // Construct Pass.
  SpireSubPass pass(passName, vboName, iboName, shader, colorScheme, state, renderType, geomVBO, geomIBO, text);

  // Add all uniforms generated above to the pass.
  for (const auto& uniform : uniforms) { pass.addUniform(uniform); }

  geom.vbos().push_back(geomVBO);
  geom.ibos().push_back(geomIBO);
  geom.passes().push_back(pass);
}

void GlyphGeom::addArrow(const Point& p1, const Point& p2, double radius, double ratio, int resolution,
  const ColorRGB& color1, const ColorRGB& color2)
{
  Point mid((p1.x() * ratio + p2.x() * (1 - ratio)), (p1.y() * ratio + p2.y() * (1 - ratio)), (p1.z() * ratio + p2.z() * (1 - ratio)));

  generateCylinder(p1, mid, radius / 6.0, radius / 6.0, resolution, color1, color2);
  generateCone(mid, p2, radius, resolution, false, color1, color2);
// generateCylinder(mid, p2, radius, 0.0, resolution, color1, color2);
}

void GlyphGeom::addSphere(const Point& p, double radius, int resolution, const ColorRGB& color)
{
  generateSphere(p, radius, resolution, color);
}

void GlyphGeom::addBox(const Point& center, Tensor& t, double scale)
{
    generateBox(center, t, scale);
}

void GlyphGeom::addEllipsoid(const Point& p, Tensor& t, Vector& scaled_eigenvals, int resolution, const ColorRGB& color)
{
  generateEllipsoid(p, t, scaled_eigenvals, resolution, color);
}

void GlyphGeom::addCylinder(const Point& p1, const Point& p2, double radius, int resolution,
                            const ColorRGB& color1, const ColorRGB& color2)
{
  generateCylinder(p1, p2, radius, radius, resolution, color1, color2);
}

void GlyphGeom::addDisk(const Point& p1, const Point& p2, double radius, int resolution,
                            const ColorRGB& color1, const ColorRGB& color2)
{
  generateDisk(p1, p2, radius, radius, resolution, color1, color2);
}

void GlyphGeom::addCone(const Point& p1, const Point& p2, double radius, int resolution,
                        bool renderBase, const ColorRGB& color1, const ColorRGB& color2)
{
  generateCone(p1, p2, radius, resolution, renderBase, color1, color2);
}

void GlyphGeom::addClippingPlane(const Point& p1, const Point& p2,
  const Point& p3, const Point& p4, double radius, int resolution,
  const ColorRGB& color1, const ColorRGB& color2)
{
  addSphere(p1, radius, resolution, color1);
  addSphere(p2, radius, resolution, color1);
  addSphere(p3, radius, resolution, color1);
  addSphere(p4, radius, resolution, color1);
  addCylinder(p1, p2, radius, resolution, color1, color2);
  addCylinder(p2, p3, radius, resolution, color1, color2);
  addCylinder(p3, p4, radius, resolution, color1, color2);
  addCylinder(p4, p1, radius, resolution, color1, color2);
}

void GlyphGeom::addPlane(const Point& p1, const Point& p2,
  const Point& p3, const Point& p4,
  const ColorRGB& color1)
{
  generatePlane(p1, p2, p3, p4, color1);
}

void GlyphGeom::addLine(const Point& p1, const Point& p2, const ColorRGB& color1, const ColorRGB& color2)
{
  generateLine(p1, p2, color1, color2);
}

void GlyphGeom::addNeedle(const Point& p1, const Point& p2, const ColorRGB& color1, const ColorRGB& color2)
{
  Point mid(0.5 * (p1.x() + p2.x()), 0.5 * (p1.y() + p2.y()), 0.5 * (p1.z() + p2.z()));
  ColorRGB endColor(color2.r(), color2.g(), color2.b(), 0.5);
  generateLine(p1, mid, color1, endColor);
  generateLine(mid, p2, color1, endColor);
}

void GlyphGeom::addPoint(const Point& p, const ColorRGB& color)
{
  generatePoint(p, color);
}

void GlyphGeom::generateCylinder(const Point& p1, const Point& p2, double radius1,
                                 double radius2, int resolution, const ColorRGB& color1,
                                 const ColorRGB& color2)
{
  double num_strips = resolution;
  if (num_strips < 0) num_strips = 20.0;
  double r1 = radius1 < 0 ? 1.0 : radius1;
  double r2 = radius2 < 0 ? 1.0 : radius2;

  //generate triangles for the cylinders.
  Vector n((p1 - p2).normal());
  Vector crx = n.getArbitraryTangent();
  Vector u = Cross(crx, n).normal();
  Vector p;
  for (int strips = 0; strips <= num_strips; strips++)
  {
    size_t offset = static_cast<size_t>(numVBOElements_);
    p = std::cos(2. * M_PI * strips / num_strips) * u +
      std::sin(2. * M_PI * strips / num_strips) * crx;
    p.normalize();
    Vector normals(((p2-p1).length() * p + (r2-r1)*n).normal());

    points_.push_back(r1 * p + Vector(p1));
    colors_.push_back(color1);
    normals_.push_back(normals);
    numVBOElements_++;
    points_.push_back(r2 * p + Vector(p2));
    colors_.push_back(color2);
    normals_.push_back(normals);
    numVBOElements_++;

    indices_.push_back(0 + offset);
    indices_.push_back(1 + offset);
    indices_.push_back(2 + offset);
    indices_.push_back(2 + offset);
    indices_.push_back(1 + offset);
    indices_.push_back(3 + offset);
  }
  for (int jj = 0; jj < 6; jj++) indices_.pop_back();
}

void GlyphGeom::generateCone(const Point& p1, const Point& p2, double radius,
                             int resolution, bool renderBase,
                             const ColorRGB& color1, const ColorRGB& color2)
{
  resolution = resolution < 0 ? 20 : resolution;
  radius = radius < 0 ? 1 : radius;

  //generate triangles for the cylinders.
  Vector n((p1 - p2).normal());
  Vector crx = n.getArbitraryTangent();
  Vector u = Cross(crx, n).normal();

  // Center of base
  size_t base_index = numVBOElements_;
  int points_per_loop = 2;
  if(renderBase)
  {
    points_.push_back(Vector(p1));
    colors_.push_back(color1);
    normals_.push_back(n);
    numVBOElements_++;
    points_per_loop = 3;
  }

  // Precalculate
  double length = (p2-p1).length();
  double strip_angle = 2. * M_PI / resolution;
  size_t offset = static_cast<size_t>(numVBOElements_);

  Vector p;

  // Add points, normals, and colors
  for (int strips = 0; strips <= resolution; strips++)
  {
    p = std::cos(strip_angle * strips) * u +
      std::sin(strip_angle * strips) * crx;
    p.normalize();
    Vector normals((length * p - radius * n).normal());

    points_.push_back(radius * p + Vector(p1));
    colors_.push_back(color1);
    normals_.push_back(normals);
    points_.push_back(Vector(p2));
    colors_.push_back(color2);
    normals_.push_back(normals);
    numVBOElements_ += 2;

    if(renderBase)
    {
      points_.push_back(radius * p + Vector(p1));
      colors_.push_back(color1);
      normals_.push_back(n);
      numVBOElements_++;
    }
  }

  // Add indices
  for (int strips = offset; strips < resolution * points_per_loop + offset; strips += points_per_loop)
  {
    indices_.push_back(strips);
    indices_.push_back(strips + 1);
    indices_.push_back(strips + points_per_loop);
    if(renderBase)
    {
      indices_.push_back(base_index);
      indices_.push_back(strips + 2);
      indices_.push_back(strips + points_per_loop + 2);
    }
  }
}

void GlyphGeom::generateDisk(const Point& p1, const Point& p2, double radius1,
                             double radius2, int resolution, const ColorRGB& color1,
                             const ColorRGB& color2)
{
  resolution = resolution < 0 ? 20 : resolution;
  radius1 = radius1 < 0 ? 1.0 : radius1;
  radius2 = radius2 < 0 ? 1.0 : radius2;

  //generate triangles for the cylinders.
  Vector n((p1 - p2).normal());
  Vector crx = n.getArbitraryTangent();
  Vector u = Cross(crx, n).normal();

  int points_per_loop = 4;

  // Add center points so flat sides can be drawn
  points_.push_back(Vector(p1));
  points_.push_back(Vector(p2));
  int p1_index = numVBOElements_;
  colors_.push_back(color1);
  normals_.push_back(n);
  numVBOElements_++;
  int p2_index = numVBOElements_;
  colors_.push_back(color2);
  normals_.push_back(-n);
  numVBOElements_++;

  // Precalculate
  double length = (p2-p1).length();
  double strip_angle = 2. * M_PI / resolution;
  size_t offset = static_cast<size_t>(numVBOElements_);

  Vector p;
  // Add points, normals, and colors
  for (int strips = 0; strips <= resolution; strips++)
  {
    p = std::cos(strip_angle * strips) * u +
      std::sin(strip_angle * strips) * crx;
    p.normalize();
    Vector normals((length * p + (radius2-radius1)*n).normal());
    points_.push_back(radius1 * p + Vector(p1));
    colors_.push_back(color1);
    normals_.push_back(normals);
    points_.push_back(radius2 * p + Vector(p2));
    colors_.push_back(color2);
    normals_.push_back(normals);

    // Points for base
    points_.push_back(radius1 * p + Vector(p1));
    colors_.push_back(color1);
    normals_.push_back(n);
    points_.push_back(radius2 * p + Vector(p2));
    colors_.push_back(color2);
    normals_.push_back(-n);
    numVBOElements_ += 4;
  }

  // Add indices
  for (int strips = offset; strips < resolution * points_per_loop + offset; strips += points_per_loop)
  {
    indices_.push_back(strips);
    indices_.push_back(strips + 1);
    indices_.push_back(strips + points_per_loop);
    indices_.push_back(strips + points_per_loop);
    indices_.push_back(strips + 1);
    indices_.push_back(strips + points_per_loop + 1);

    // Render base
    indices_.push_back(p1_index);
    indices_.push_back(strips + 2);
    indices_.push_back(strips + points_per_loop + 2);
    indices_.push_back(strips + 3);
    indices_.push_back(p2_index);
    indices_.push_back(strips + points_per_loop + 3);
  }
}

void GlyphGeom::generateSphere(const Point& center, double radius, int resolution, const ColorRGB& color)
{
  double num_strips = resolution;
  if (num_strips < 0) num_strips = 20.0;
  double r = radius < 0 ? 1.0 : radius;
  Vector pp1, pp2;
  double theta_inc = /*2. */ M_PI / num_strips, phi_inc = 0.5 * M_PI / num_strips;

  //generate triangles for the spheres
  for (double phi = 0.; phi <= M_PI - phi_inc; phi += phi_inc)
  {
    for (double theta = 0.; theta <= 2. * M_PI; theta += theta_inc)
    {
      uint32_t offset = static_cast<uint32_t>(numVBOElements_);
      pp1 = Vector(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
      pp2 = Vector(sin(theta) * cos(phi + phi_inc), sin(theta) * sin(phi + phi_inc), cos(theta));

      normals_.push_back(pp1);
      normals_.push_back(pp2);
      pp1 *= r;
      pp2 *= r;
      points_.push_back(pp1 + Vector(center));
      colors_.push_back(color);
      numVBOElements_++;
      points_.push_back(pp2 + Vector(center));
      colors_.push_back(color);
      numVBOElements_++;

      //preserve vertex ordering for double sided rendering
      int v1 = 1, v2 = 2;
      if(theta < M_PI)
      {
        v1 = 2;
        v2 = 1;
      }

      indices_.push_back(0 + offset);
      indices_.push_back(v1 + offset);
      indices_.push_back(v2 + offset);
      indices_.push_back(v2 + offset);
      indices_.push_back(v1 + offset);
      indices_.push_back(3 + offset);
    }
    for (int jj = 0; jj < 6; jj++) indices_.pop_back();
  }
}

void GlyphGeom::generateBox(const Point& center, Tensor& t, double scale)
{
    /**
    std::vector<QuadStrip> quadstrips;
    double eigval1, eigval2, eigval3;
    t.get_eigenvalues(eigval1, eigval2, eigval3);

    double half_x_side = eigval1 * 0.5 * scale;
    double half_y_side = eigval2 * 0.5 * scale;
    double half_z_side = eigval3 * 0.5 * scale;

    std::cout << "box:\nx: " << half_x_side << "\ny: " << half_y_side << "\nz: " << half_z_side << "\n\n";

    Transform trans;
    Transform rotate;
//    generateTransforms(center, t, trans, rotate);

    uint32_t offset = static_cast<uint32_t>(numVBOElements_);
    //Draw the Box
//    Point p1 = trans * Point(-half_x_side, half_y_side, half_z_side);
//    Point p2 = trans * Point(-half_x_side, half_y_side, -half_z_side);
//    Point p3 = trans * Point(half_x_side, half_y_side, half_z_side);
//    Point p4 = trans * Point(half_x_side, half_y_side, -half_z_side);
//
//    Point p5 = trans * Point(-half_x_side, -half_y_side, half_z_side);
//    Point p6 = trans * Point(-half_x_side, -half_y_side, -half_z_side);
//    Point p7 = trans * Point(half_x_side, -half_y_side, half_z_side);
//    Point p8 = trans * Point(half_x_side, -half_y_side, -half_z_side);

    // Make vectors
    Vector v_corner1 = Point(-half_x_side, half_y_side, half_z_side);
    Vector v_corner2 = Point(-half_x_side, half_y_side, -half_z_side);
    Vector v_corner3 = Point(half_x_side, half_y_side, half_z_side);
    Vector v_corner4 = Point(half_x_side, half_y_side, -half_z_side);

    Vector v_corner5 = Point(-half_x_side, -half_y_side, half_z_side);
    Vector v_corner6 = Point(-half_x_side, -half_y_side, -half_z_side);
    Vector v_corner7 = Point(half_x_side, -half_y_side, half_z_side);
    Vector v_corner8 = Point(half_x_side, -half_y_side, -half_z_side);

    Vector v_plane1 = rotate * Vector(half_x_side, 0, 0);
    Vector v_plane2 = rotate * Vector(0, half_y_side, 0);
    Vector v_plane3 = rotate * Vector(0, 0, half_z_side);

    Vector v_plane4 = rotate * Vector(-half_x_side, 0, 0);
    Vector v_plane5 = rotate * Vector(0, -half_y_side, 0);
    Vector v_plane6 = rotate * Vector(0, 0, -half_z_side);

    // Add corner points to list
    points_.push_back(v_plane1 + Vector(center));
    points_.push_back(v_plane2 + Vector(center));
    points_.push_back(v_plane3 + Vector(center));
    points_.push_back(v_plane4 + Vector(center));
    points_.push_back(v_plane5 + Vector(center));
    points_.push_back(v_plane6 + Vector(center));
    points_.push_back(v_pp1 + Vector(center));
    points_.push_back(v_pp2 + Vector(center));
    points_.push_back(v_pp3 + Vector(center));
    points_.push_back(v_pp4 + Vector(center));
    points_.push_back(v_pp5 + Vector(center));
    points_.push_back(v_pp6 + Vector(center));
    points_.push_back(v_pp7 + Vector(center));
    points_.push_back(v_pp8 + Vector(center));

    // Add indices
    indices_.push_back(0 + offset);
    indices_.push_back(1 + offset);
    indices_.push_back(2 + offset);
    quadstrip1.push_back((v_corner7, v_plane1));
    quadstrip1.push_back((v_corner8, v_plane1));
    quadstrip1.push_back((v_corner3, v_plane1));
    quadstrip1.push_back((v_corner4, v_plane1));

    Vector v1 = rotate * Vector(half_x_side, 0, 0);
    Vector v2 = rotate * Vector(0, half_y_side, 0);
    Vector v3 = rotate * Vector(0, 0, half_z_side);

    Vector v4 = rotate * Vector(-half_x_side, 0, 0);
    Vector v5 = rotate * Vector(0, -half_y_side, 0);
    Vector v6 = rotate * Vector(0, 0, -half_z_side);

    QuadStrip quadstrip1;
    QuadStrip quadstrip2;
    QuadStrip quadstrip3;
    QuadStrip quadstrip4;
    QuadStrip quadstrip5;
    QuadStrip quadstrip6;

    // +X
    quadstrip1.push_back(std::make_pair(p7, v1));
    quadstrip1.push_back(std::make_pair(p8, v1));
    quadstrip1.push_back(std::make_pair(p3, v1));
    quadstrip1.push_back(std::make_pair(p4, v1));

    // +Y
    quadstrip2.push_back(std::make_pair(p3, v2));
    quadstrip2.push_back(std::make_pair(p4, v2));
    quadstrip2.push_back(std::make_pair(p1, v2));
    quadstrip2.push_back(std::make_pair(p2, v2));

    // +Z
    quadstrip3.push_back(std::make_pair(p5, v3));
    quadstrip3.push_back(std::make_pair(p7, v3));
    quadstrip3.push_back(std::make_pair(p1, v3));
    quadstrip3.push_back(std::make_pair(p3, v3));

    // -X
    quadstrip4.push_back(std::make_pair(p1, v4));
    quadstrip4.push_back(std::make_pair(p2, v4));
    quadstrip4.push_back(std::make_pair(p5, v4));
    quadstrip4.push_back(std::make_pair(p6, v4));

    // -Y
    quadstrip5.push_back(std::make_pair(p5, v5));
    quadstrip5.push_back(std::make_pair(p6, v5));
    quadstrip5.push_back(std::make_pair(p7, v5));
    quadstrip5.push_back(std::make_pair(p8, v5));

    // -Z
    quadstrip6.push_back(std::make_pair(p2, v6));
    quadstrip6.push_back(std::make_pair(p4, v6));
    quadstrip6.push_back(std::make_pair(p6, v6));
    quadstrip6.push_back(std::make_pair(p8, v6));

    quadstrips.push_back(quadstrip1);
    quadstrips.push_back(quadstrip2);
    quadstrips.push_back(quadstrip3);
    quadstrips.push_back(quadstrip4);
    quadstrips.push_back(quadstrip5);
    quadstrips.push_back(quadstrip6);
     **/
}

void GlyphGeom::generateEllipsoid(const Point& center, Tensor& t, Vector &scaled_eigenvals, int resolution, const ColorRGB& color)
{
    Vector eig_vec1, eig_vec2, eig_vec3;
    t.get_eigenvectors(eig_vec1, eig_vec2, eig_vec3);

    // Scale to eigen values
    eig_vec1 *= scaled_eigenvals.x();
    eig_vec2 *= scaled_eigenvals.y();
    eig_vec3 *= scaled_eigenvals.z();

    int nu = resolution + 1;
    //    int nv = resolution;

    // Half ellipsoid criteria.
    //  if (half == -1) start = M_PI / 2.0;
    //  if (half == 1) stop = M_PI / 2.0;
    //  if (half != 0) nv /= 2;

    // Should only happen when doing half ellipsoids.
    //  if (nv < 2) nv = 2;

    SinCosTable tab1(nu, 0, 2 * M_PI);
    SinCosTable tab2(resolution, 0, M_PI);

    // Draw the ellipsoid
    for (int v = 0; v<resolution - 1; v++)
      {
        double nr1 = tab2.sin(v + 1);
        double nr2 = tab2.sin(v);

        double nz1 = tab2.cos(v + 1);
        double nz2 = tab2.cos(v);

        for (int u = 0; u<nu; u++)
          {
            uint32_t offset = static_cast<uint32_t>(numVBOElements_);
            double nx = tab1.sin(u);
            double ny = tab1.cos(u);

            double x1 = nr1 * nx;
            double y1 = nr1 * ny;
            double z1 = nz1;

            double x2 = nr2 * nx;
            double y2 = nr2 * ny;
            double z2 = nz2;

            // Rotate points
            Vector v_p1 = Vector(eig_vec1[0] * x1 + eig_vec2[0] * y1 + eig_vec3[0] * z1,
                                 eig_vec1[1] * x1 + eig_vec2[1] * y1 + eig_vec3[1] * z1,
                                 eig_vec1[2] * x1 + eig_vec2[2] * y1 + eig_vec3[2] * z1);

            Vector v_p2 = Vector(eig_vec1[0] * x2 + eig_vec2[0] * y2 + eig_vec3[0] * z2,
                                 eig_vec1[1] * x2 + eig_vec2[1] * y2 + eig_vec3[1] * z2,
                                 eig_vec1[2] * x2 + eig_vec2[2] * y2 + eig_vec3[2] * z2);

            // Transorm points and add to points list
            points_.push_back(v_p1 + Vector(center));
            points_.push_back(v_p2 + Vector(center));

            // Add normals
            normals_.push_back(v_p1);
            normals_.push_back(v_p2);

            // Add color vectors from parameters
            colors_.push_back(color);
            colors_.push_back(color);

            numVBOElements_ += 2;

            indices_.push_back(0 + offset);
            indices_.push_back(1 + offset);
            indices_.push_back(2 + offset);
            indices_.push_back(2 + offset);
            indices_.push_back(1 + offset);
            indices_.push_back(3 + offset);
          }
        for(int jj = 0; jj < 6; jj++) indices_.pop_back();
      }
}

void GlyphGeom::generateLine(const Point& p1, const Point& p2, const ColorRGB& color1, const ColorRGB& color2)
{
  points_.push_back(Vector(p1));
  colors_.push_back(color1);
  indices_.push_back(lineIndex_);
  ++lineIndex_;
  points_.push_back(Vector(p2));
  colors_.push_back(color2);
  indices_.push_back(lineIndex_);
  ++lineIndex_;
  ++numVBOElements_;
}

void GlyphGeom::generatePoint(const Point& p, const ColorRGB& color)
{
  points_.push_back(Vector(p));
  colors_.push_back(color);
  indices_.push_back(lineIndex_);
  ++lineIndex_;
  ++numVBOElements_;
}

void GlyphGeom::generatePlane(const Point& p1, const Point& p2,
  const Point& p3, const Point& p4, const ColorRGB& color)
{
  points_.push_back(Vector(p1));
  points_.push_back(Vector(p2));
  points_.push_back(Vector(p3));
  points_.push_back(Vector(p4));
  colors_.push_back(color);
  colors_.push_back(color);
  colors_.push_back(color);
  colors_.push_back(color);
  Vector n;
  n = Cross(p2 - p1, p4 - p1).normal();
  normals_.push_back(n);
  n = Cross(p3 - p2, p1 - p2).normal();
  normals_.push_back(n);
  n = Cross(p4 - p3, p2 - p3).normal();
  normals_.push_back(n);
  n = Cross(p1 - p4, p3 - p4).normal();
  normals_.push_back(n);
  indices_.push_back(0 + numVBOElements_);
  indices_.push_back(1 + numVBOElements_);
  indices_.push_back(2 + numVBOElements_);
  indices_.push_back(2 + numVBOElements_);
  indices_.push_back(3 + numVBOElements_);
  indices_.push_back(0 + numVBOElements_);
  numVBOElements_ += 4;
}

// Addarrow from SCIRun 4
void GlyphGeom::addArrow(const Point& center, const Vector& t,
                         double radius, double length, int nu, int nv)
{
  std::vector<QuadStrip> quadstrips;
  double ratio = 2.0;
  Transform trans;
  Transform rotate;
  generateTransforms(center, t, trans, rotate);

  Vector offset = rotate * Vector(0,0,1);
  offset.safe_normalize();
  offset *= length * ratio;

  generateCylinder(center, t, radius/10.0, radius/10.0, length*ratio, nu, nv, quadstrips);
  generateCylinder(center+offset, t, radius, 0.0, length, nu, nv, quadstrips);

  // add strips to the object
}

// from SCIRun 4
void GlyphGeom::addBox(const Point& center, const Vector& t,
                       double x_side, double y_side, double z_side)
{
  std::vector<QuadStrip> quadstrips;
  generateBox(center, t, x_side, y_side, z_side, quadstrips);

  // add strips to object
}

// from SCIRun 4
void GlyphGeom::addCylinder(const Point& center, const Vector& t,
                            double radius1, double length, int nu, int nv)
{
  std::vector<QuadStrip> quadstrips;
  generateCylinder(center, t, radius1, radius1, length, nu, nv, quadstrips);

  // add the strips to the object
}

// from SCIRun 4
void GlyphGeom::addSphere(const Point& center, double radius,
                          int nu, int nv, int half)
{
  std::vector<QuadStrip> quadstrips;
  generateEllipsoid(center, Vector(0, 0, 1), radius, nu, nv, half, quadstrips);

  // add strips to the object

}

// Generate cylinder from SCIRun 4
void GlyphGeom::generateCylinder(const Point& center, const Vector& t, double radius1,
                                 double radius2, double length, int nu, int nv,
                                 std::vector<QuadStrip>& quadstrips)
{
  nu++; //Bring nu to expected value for shape

  if (nu > 20) nu = 20;
  if (nv == 0) nv = 20;
  SinCosTable& tab1 = tables_[nu];

  Transform trans;
  Transform rotate;
  generateTransforms(center, t, trans, rotate);

  // Draw the cylinder
  double dz = length / static_cast<float>(nv);
  double dr = (radius2 - radius1) / static_cast<float>(nv);

  for (int v = 0; v<nv; v++)
  {
    double z1 = dz * static_cast<float>(v);
    double z2 = z1 + dz;

    double r1 = radius1 + dr * static_cast<float>(v);
    double r2 = r1 + dr;

    QuadStrip quadstrip;

    for (int u = 0; u<nu; u++)
    {
      double nx = tab1.sin(u);
      double ny = tab1.cos(u);

      double x1 = r1 * nx;
      double y1 = r1 * ny;

      double x2 = r2 * nx;
      double y2 = r2 * ny;

      double nx1 = length * nx;
      double ny1 = length * ny;

      Point p1 = trans * Point(x1, y1, z1);
      Point p2 = trans * Point(x2, y2, z2);

      Vector v1 = rotate * Vector(nx1, ny1, -dr);
      v1.safe_normalize();

      quadstrip.push_back(std::make_pair(p1, v1));
      quadstrip.push_back(std::make_pair(p2, v1));
    }

    quadstrips.push_back(quadstrip);
  }

}

// generate ellipsoid from SCIRun 4
void GlyphGeom::generateEllipsoid(const Point& center, const Vector& t, double scales,
                                  int nu, int nv, int half, std::vector<QuadStrip>& quadstrips)
{
  nu++; //Bring nu to expected value for shape.

  double start = 0, stop =  M_PI;

  // Half ellipsoid criteria.
  if (half == -1) start = M_PI / 2.0;
  if (half == 1) stop = M_PI / 2.0;
  if (half != 0) nv /= 2;

  // Should only happen when doing half ellipsoids.
  if (nv < 2) nv = 2;

  SinCosTable tab1(nu, 0, 2 * M_PI);
  SinCosTable tab2(nv, start, stop);

  Transform trans;
  Transform rotate;
  generateTransforms(center, t, trans, rotate);

  trans.post_scale(Vector(1.0, 1.0, 1.0) * scales);
  rotate.post_scale(Vector(1.0, 1.0, 1.0) / scales);

  // Draw the ellipsoid
  for (int v = 0; v<nv - 1; v++)
  {
    double nr1 = tab2.sin(v + 1);
    double nr2 = tab2.sin(v);

    double nz1 = tab2.cos(v + 1);
    double nz2 = tab2.cos(v);

    QuadStrip quadstrip;

    for (int u = 0; u<nu; u++)
    {
      double nx = tab1.sin(u);
      double ny = tab1.cos(u);

      double x1 = nr1 * nx;
      double y1 = nr1 * ny;
      double z1 = nz1;

      double x2 = nr2 * nx;
      double y2 = nr2 * ny;
      double z2 = nz2;

      Point p1 = trans * Point(x1, y1, z1);
      Point p2 = trans * Point(x2, y2, z2);

      Vector v1 = rotate * Vector(x1, y1, z1);
      Vector v2 = rotate * Vector(x2, y2, z2);

      v1.safe_normalize();
      v2.safe_normalize();

      quadstrip.push_back(std::make_pair(p1, v1));
      quadstrip.push_back(std::make_pair(p2, v2));
    }

    quadstrips.push_back(quadstrip);
  }
}

//generate box from SCIRun 4
void GlyphGeom::generateBox(const Point& center, const Vector& t, double x_side, double y_side,
                            double z_side, std::vector<QuadStrip>& quadstrips)
{
  double half_x_side = x_side * 0.5;
  double half_y_side = y_side * 0.5;
  double half_z_side = z_side * 0.5;

  Transform trans;
  Transform rotate;
  generateTransforms(center, t, trans, rotate);

  //Draw the Box
  Point p1 = trans * Point(-half_x_side, half_y_side, half_z_side);
  Point p2 = trans * Point(-half_x_side, half_y_side, -half_z_side);
  Point p3 = trans * Point(half_x_side, half_y_side, half_z_side);
  Point p4 = trans * Point(half_x_side, half_y_side, -half_z_side);

  Point p5 = trans * Point(-half_x_side, -half_y_side, half_z_side);
  Point p6 = trans * Point(-half_x_side, -half_y_side, -half_z_side);
  Point p7 = trans * Point(half_x_side, -half_y_side, half_z_side);
  Point p8 = trans * Point(half_x_side, -half_y_side, -half_z_side);

  Vector v1 = rotate * Vector(half_x_side, 0, 0);
  Vector v2 = rotate * Vector(0, half_y_side, 0);
  Vector v3 = rotate * Vector(0, 0, half_z_side);

  Vector v4 = rotate * Vector(-half_x_side, 0, 0);
  Vector v5 = rotate * Vector(0, -half_y_side, 0);
  Vector v6 = rotate * Vector(0, 0, -half_z_side);

  QuadStrip quadstrip1;
  QuadStrip quadstrip2;
  QuadStrip quadstrip3;
  QuadStrip quadstrip4;
  QuadStrip quadstrip5;
  QuadStrip quadstrip6;

  // +X
  quadstrip1.push_back(std::make_pair(p7, v1));
  quadstrip1.push_back(std::make_pair(p8, v1));
  quadstrip1.push_back(std::make_pair(p3, v1));
  quadstrip1.push_back(std::make_pair(p4, v1));

  // +Y
  quadstrip2.push_back(std::make_pair(p3, v2));
  quadstrip2.push_back(std::make_pair(p4, v2));
  quadstrip2.push_back(std::make_pair(p1, v2));
  quadstrip2.push_back(std::make_pair(p2, v2));

  // +Z
  quadstrip3.push_back(std::make_pair(p5, v3));
  quadstrip3.push_back(std::make_pair(p7, v3));
  quadstrip3.push_back(std::make_pair(p1, v3));
  quadstrip3.push_back(std::make_pair(p3, v3));

  // -X
  quadstrip4.push_back(std::make_pair(p1, v4));
  quadstrip4.push_back(std::make_pair(p2, v4));
  quadstrip4.push_back(std::make_pair(p5, v4));
  quadstrip4.push_back(std::make_pair(p6, v4));

  // -Y
  quadstrip5.push_back(std::make_pair(p5, v5));
  quadstrip5.push_back(std::make_pair(p6, v5));
  quadstrip5.push_back(std::make_pair(p7, v5));
  quadstrip5.push_back(std::make_pair(p8, v5));

  // -Z
  quadstrip6.push_back(std::make_pair(p2, v6));
  quadstrip6.push_back(std::make_pair(p4, v6));
  quadstrip6.push_back(std::make_pair(p6, v6));
  quadstrip6.push_back(std::make_pair(p8, v6));

  quadstrips.push_back(quadstrip1);
  quadstrips.push_back(quadstrip2);
  quadstrips.push_back(quadstrip3);
  quadstrips.push_back(quadstrip4);
  quadstrips.push_back(quadstrip5);
  quadstrips.push_back(quadstrip6);
}

// from SCIRun 4
void GlyphGeom::generateTransforms(const Point& center, const Vector& normal,
                                   Transform& trans, Transform& rotate)
{
  Vector axis = normal;

  axis.normalize();

  Vector z(0, 0, 1), zrotaxis;

  if((Abs(axis.x()) + Abs(axis.y())) < 1.e-5)
  {
    // Only x-z plane...
    zrotaxis = Vector(0, 1, 0);
  }
  else
  {
    zrotaxis = Cross(axis, z);
    zrotaxis.normalize();
  }

  double cangle = Dot(z, axis);
  double zrotangle = -acos(cangle);

  rotate.post_rotate(zrotangle, zrotaxis);
}
