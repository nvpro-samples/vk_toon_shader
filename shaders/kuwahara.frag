// by Jan Eric Kyprianidis <www.kyprianidis.com>

#version 450

// clang-format off
layout(set = 0, binding = 0) uniform sampler2D iChannel0;  // Normal + depth
layout(push_constant) uniform  params_ {int radius;};
// clang-format on

layout(location = 0) in vec2 fragCoord;
layout(location = 0) out vec4 fragColor;


void main(void)
{
  vec2  src_size = vec2(textureSize(iChannel0, 0));
  vec2  uv       = gl_FragCoord.xy / src_size;
  float n        = float((radius + 1) * (radius + 1));

  float alpha = texture(iChannel0, uv).a;

  vec3 m[4];
  vec3 s[4];
  for(int k = 0; k < 4; ++k)
  {
    m[k] = vec3(0.0);
    s[k] = vec3(0.0);
  }

  for(int j = -radius; j <= 0; ++j)
  {
    for(int i = -radius; i <= 0; ++i)
    {
      vec3 c = texture(iChannel0, uv + vec2(i, j) / src_size).rgb;
      m[0] += c;
      s[0] += c * c;
    }
  }

  for(int j = -radius; j <= 0; ++j)
  {
    for(int i = 0; i <= radius; ++i)
    {
      vec3 c = texture(iChannel0, uv + vec2(i, j) / src_size).rgb;
      m[1] += c;
      s[1] += c * c;
    }
  }

  for(int j = 0; j <= radius; ++j)
  {
    for(int i = 0; i <= radius; ++i)
    {
      vec3 c = texture(iChannel0, uv + vec2(i, j) / src_size).rgb;
      m[2] += c;
      s[2] += c * c;
    }
  }

  for(int j = 0; j <= radius; ++j)
  {
    for(int i = -radius; i <= 0; ++i)
    {
      vec3 c = texture(iChannel0, uv + vec2(i, j) / src_size).rgb;
      m[3] += c;
      s[3] += c * c;
    }
  }


  float min_sigma2 = 1e+2;
  for(int k = 0; k < 4; ++k)
  {
    m[k] /= n;
    s[k] = abs(s[k] / n - m[k] * m[k]);

    float sigma2 = s[k].r + s[k].g + s[k].b;
    if(sigma2 < min_sigma2)
    {
      min_sigma2 = sigma2;
      fragColor  = vec4(m[k], alpha);
    }
  }
}
