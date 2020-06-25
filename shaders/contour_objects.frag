#version 450

// clang-format off
layout(set = 0, binding = 0) uniform sampler2D iChannel0;  // Object ID in Z
// clang-format on

layout(push_constant) uniform params_
{
  int contourMethod;
};


layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;


vec4 objectContour()
{
  ivec2 size       = textureSize(iChannel0, 0);
  ivec2 texelCoord = ivec2(vec2(size) * fragCoord.st);

  int A = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, +1.0)).w);  //  +---+---+---+
  int B = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, +1.0)).w);  //  | A | B | C |
  int C = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, +1.0)).w);  //  +---+---+---+
  int D = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, +0.0)).w);  //  | D | X | E |
  int X = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, +0.0)).w);  //  +---+---+---+
  int E = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, +0.0)).w);  //  | F | G | H |
  int F = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(-1.0, -1.0)).w);  //  +---+---+---+
  int G = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+0.0, -1.0)).w);
  int H = floatBitsToInt(texelFetchOffset(iChannel0, texelCoord, 0, ivec2(+1.0, -1.0)).w);


  switch(contourMethod)
  {
    case 0:  // smaller
      if(X < A || X < B || X < C || X < D || X < E || X < F || X < G || X < H)
      {
        return vec4(1);
      }
      break;
    case 1:  // bigger
      if(X > A || X > B || X > C || X > D || X > E || X > F || X > G || X > H)
      {
        return vec4(1);
      }
      break;
    case 2:  // thicker
      if(X != A || X != B || X != C || X != D || X != E || X != F || X != G || X != H)
      {
        return vec4(1);
      }
    case 3:  // different
      return vec4((int(X != A) + int(X != C) + int(X != F) + int(X != H)) * (1. / 6.)
                  + (int(X != B) + int(X != D) + int(X != E) + int(X != G)) * (1. / 3.));

      break;
  }

  return vec4(0);
}

void main()
{
  fragColor = objectContour();
}
