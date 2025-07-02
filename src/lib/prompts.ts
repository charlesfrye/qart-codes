export const SAMPLE_PROMPTS = [
    // // aim for an equal distribution across prompt types
    // artist styles
    "Cy Twombly",
    "Mondrian",
    "Sol LeWitt",
    "Rauschenberg",
    // comma-separated
    "Penguins have a picnic on the savannah, beautiful landscape, sharp focus, masterpiece, 8k, intricate artwork, nature photography, hyper detailed, highly detailed, HD, 4K",
    "neon green cubes, rendered in blender, trending on artstation, deep colors, cyberpunk aesthetic, striking contrast, hyperrealistic",
    "crystalline structures, volumetric lighting, rendered in octane, photorealistic textures, prismatic reflections, studio lighting setup",
    "paper cut art style, layered shadows, pastel color palette, handcrafted textures, stop motion animation feel, soft ambient lighting",
    // ekphrasis (prose description)
    "An abstract art piece inspired by dense forest foliage, featuring intricate patterns of leaves, branches, and flowers in vibrant shades of green, yellow, and gold. The design includes layered textures and soft gradients to create depth and a harmonious balance of details, illuminated by scattered rays of sunlight filtering through a canopy.",
    "A dense jungle scene filled with towering trees, intertwining vines, and an abundance of foliage. The composition features detailed textures of leaves, moss, and bark, with sunlight streaming through the canopy to create dappled light patterns on the forest floor.",
    "A surreal underwater coral reef filled with intricate details of colorful corals, seaweed, and marine life. The scene features vibrant fish, glowing bioluminescent plants, and rich textures, with light filtering from the water's surface creating soft, scattered patterns.",
    "A lush, densely packed bouquet of flowers featuring a variety of species in full bloom. The flowers display intricate petal textures and patterns in a vibrant mix of colors, with soft shadows and subtle gradients creating depth and harmony.",
    "A whimsical forest scene with fantastical trees, glowing mushrooms, and colorful wildflowers. The image is rich in detail, with textured bark, intricate leaves, and scattered beams of light breaking through the trees, casting soft, glowing patterns on the ground.",
  ];

  export const getRandomPrompts = (count = 6) => {
    const shuffled = [...SAMPLE_PROMPTS].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  };
