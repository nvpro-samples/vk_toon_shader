// by Jan Eric Kyprianidis <www.kyprianidis.com>

#version 450

layout(set = 0, binding = 0) uniform sampler2D src;
//layout(push_constant) uniform params_
//{
//  float sigma;
//};

layout(location = 0) out vec4 dst;
layout(location = 0) in vec2 fragCoord;

const float sigma = 2.;

void main(void)
{
  vec2  src_size  = vec2(textureSize(src, 0));
  vec2  uv        = gl_FragCoord.xy / src_size;
  float twoSigma2 = 2.0 * sigma * sigma;
  int   halfWidth = int(ceil(2.0 * sigma));

  vec3  sum  = vec3(0.0);
  float norm = 0.0;
  if(halfWidth > 0)
  {
    for(int i = -halfWidth; i <= halfWidth; ++i)
    {
      for(int j = -halfWidth; j <= halfWidth; ++j)
      {
        float d      = length(vec2(i, j));
        float kernel = exp(-d * d / twoSigma2);
        vec3  c      = texture(src, uv + vec2(i, j) / src_size).rgb;
        sum += kernel * c;
        norm += kernel;
      }
    }
  }
  else
  {
    sum  = texture(src, uv).rgb;
    norm = 1.0;
  }
  dst = vec4(sum / norm, 0);
}
