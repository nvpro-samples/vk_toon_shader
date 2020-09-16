#version 450

// https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
// https://aras-p.info/texts/CompactNormalStorage.html
vec2 encode(vec3 n)
{
  float p = sqrt(n.z * 8 + 8);
  return vec2(n.xy / p + 0.5);
}

vec3 decode(vec2 enc)
{
  vec2  fenc = enc * 4 - 2;
  float f    = dot(fenc, fenc);
  float g    = sqrt(1 - f / 4);
  vec3  n;
  n.xy = fenc * g;
  n.z  = 1 - f / 2;
  return n;
}


layout(set = 0, binding = 0) uniform sampler2D iChannel0;  // Normal + depth

layout(set = 0, binding = 1) buffer zValues
{
  int minmax[2];  // zNear and zFar, from the compute shader deapthminmax.comp
};

layout(push_constant) uniform params_
{
  float NormalDiffCoeff;
  float DepthDiffCoeff;
};


layout(location = 0) in vec2 fragCoord;
layout(location = 0) out float fragColor;


float Fdepth(in float Z, in float zNear, in float zFar)
{
  return abs((1. / Z - 1. / zNear) / ((1. / zFar) - (1. / zNear)));
}

float FNdepth(in float Z, in float zNear, in float zFar)
{
  return (Z - zNear) / (zFar - zNear);
}

float Gradient(ivec2 texelCoord, float zNear, float zFar)
{
  vec4 A = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, +1.0));  //  +---+---+---+
  vec4 B = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, +1.0));  //  | A | B | C |
  vec4 C = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, +1.0));  //  +---+---+---+
  vec4 D = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, +0.0));  //  | D | X | E |
  vec4 X = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, +0.0));  //  +---+---+---+
  vec4 E = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, +0.0));  //  | F | G | H |
  vec4 F = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, -1.0));  //  +---+---+---+
  vec4 G = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, -1.0));
  vec4 H = texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, -1.0));

  // Don't sample background
  int objId = floatBitsToInt(X.w);
  if(X.z < 0.0001 || objId == 0)
    return 0;

  vec3 An = decode(A.xy);
  vec3 Bn = decode(B.xy);
  vec3 Cn = decode(C.xy);
  vec3 Dn = decode(D.xy);
  vec3 Xn = decode(X.xy);
  vec3 En = decode(E.xy);
  vec3 Fn = decode(F.xy);
  vec3 Gn = decode(G.xy);
  vec3 Hn = decode(H.xy);

  // Normal Gradient
  float Ngrad = 0;
  {
    // compute length of gradient using Sobel/Kroon operator
    const float k0     = 17. / 23.75;
    const float k1     = 61. / 23.75;
    const vec3  grad_y = k0 * An + k1 * Bn + k0 * Cn - k0 * Fn - k1 * Gn - k0 * Hn;
    const vec3  grad_x = k0 * Cn + k1 * En + k0 * Hn - k0 * An - k1 * Dn - k0 * Fn;
    const float g      = length(grad_x) + length(grad_y);

    Ngrad = smoothstep(2.f, 3.f, g * NormalDiffCoeff);  //!! magic
  }

  // Depth Gradient
  float Dgrad = 0;
  {
    // https://www.cs.princeton.edu/courses/archive/fall00/cs597b/papers/saito90.pdf
    A.z = Fdepth(A.z, zNear, zFar);
    B.z = Fdepth(B.z, zNear, zFar);
    C.z = Fdepth(C.z, zNear, zFar);
    D.z = Fdepth(D.z, zNear, zFar);
    E.z = Fdepth(E.z, zNear, zFar);
    F.z = Fdepth(F.z, zNear, zFar);
    G.z = Fdepth(G.z, zNear, zFar);
    H.z = Fdepth(H.z, zNear, zFar);
    X.z = Fdepth(X.z, zNear, zFar);

    float g = (abs(A.z + 2 * B.z + C.z - F.z - 2 * G.z - H.z) + abs(C.z + 2 * E.z + H.z - A.z - 2 * D.z - F.z)) / 8.0;
    float l = (8 * X.z - A.z - B.z - C.z - D.z - E.z - F.z - G.z - H.z) / 3.0;

    Dgrad = (l + g) * DepthDiffCoeff;
    Dgrad = smoothstep(0.03f, 0.1f, Dgrad);  // !magic values
  }


  return Ngrad + Dgrad;
}

void main()
{
  ivec2 size       = textureSize(iChannel0, 0);
  ivec2 texelCoord = ivec2(vec2(size) * fragCoord.st);

  float zNear = intBitsToFloat(minmax[0]);
  float zFar  = intBitsToFloat(minmax[1]);

  fragColor = Gradient(texelCoord, zNear, zFar);
}
