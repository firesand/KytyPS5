#version 450
layout(location = 0) flat in vec4 parameter0;
layout(location = 1) flat in vec4 parameter1;
layout(location = 2) flat in vec4 parameter2;
layout(location = 3) flat in vec4 parameter3;
layout(location = 0) out vec4 color;
uint mixWords(uint hash, vec4 value) {
    uvec4 words = floatBitsToUint(value);
    for (int i = 0; i < 4; ++i) hash = (hash ^ words[i]) * 16777619u;
    return hash;
}
void main() {
    // Observe all four parameter vectors without arithmetic on possibly NaN
    // attributes. Flat interpolation observes the provoking vertex only.
    uint hash = mixWords(mixWords(mixWords(mixWords(2166136261u,
        parameter0), parameter1), parameter2), parameter3);
    // Fold high bits down before the RGB24 projection; otherwise input bytes
    // 24..31 could not affect the low24 bits of an xor/multiply-only hash.
    hash ^= hash >> 16;
    hash *= 2246822519u;
    hash ^= hash >> 13;
    color = vec4(vec3(uvec3(hash, hash >> 8, hash >> 16) & 255u) / 255.0, 1.0);
}
