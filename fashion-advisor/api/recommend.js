const COLOR_HARMONY = {
  neutral: ["black", "white", "gray", "beige", "navy", "cream"],
  warm: ["red", "orange", "yellow", "coral", "peach", "terracotta"],
  cool: ["blue", "green", "purple", "teal", "mint", "lavender"],
  pastel: ["pink", "lavender", "mint", "baby blue", "peach", "pale yellow"]
};

const OCCASION_STYLES = {
  casual: { tops: ["t-shirt", "hoodie", "sweater", "casual shirt"], bottoms: ["jeans", "casual pants", "shorts"], shoes: ["sneakers", "loafers", "sandals"] },
  business: { tops: ["button-down", "blazer", "dress shirt", "blouse"], bottoms: ["dress pants", "slacks", "skirt"], shoes: ["oxfords", "heels", "loafers"] },
  formal: { tops: ["dress shirt", "tuxedo shirt", "blazer"], bottoms: ["dress pants", "tuxedo pants", "dress skirt"], shoes: ["dress shoes", "heels"] },
  date: { tops: ["nice shirt", "blouse", "elegant top"], bottoms: ["nice pants", "jeans", "skirt"], shoes: ["heels", "dress shoes", "nice sneakers"] },
  workout: { tops: ["sports bra", "workout shirt", "tank top"], bottoms: ["workout pants", "shorts", "leggings"], shoes: ["running shoes", "training shoes"] },
  party: { tops: ["party top", "elegant blouse", "dressy shirt"], bottoms: ["dress pants", "party skirt", "dressy jeans"], shoes: ["heels", "dress shoes", "party sandals"] }
};

const SEASONAL = {
  spring: ["light layers", "cardigans", "jackets", "pastels", "floral"],
  summer: ["shorts", "tanks", "light fabrics", "bright colors", "sandals"],
  fall: ["layers", "sweaters", "jackets", "earth tones", "boots"],
  winter: ["heavy coats", "scarves", "boots", "dark colors", "layers"]
};

export default function handler(req, res) {
  // Handle CORS
  if (req.method === "OPTIONS") {
    return res.status(200).setHeader("Access-Control-Allow-Origin", "*").setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS").setHeader("Access-Control-Allow-Headers", "Content-Type").send("");
  }

  const { occasion = "casual", season = "spring", style = "modern" } = req.body || {};

  const baseItems = OCCASION_STYLES[occasion.toLowerCase()] || OCCASION_STYLES.casual;
  const seasonalColors = SEASONAL[season.toLowerCase()] || SEASONAL.spring;
  
  const outfit = {
    occasion: occasion.charAt(0).toUpperCase() + occasion.slice(1),
    season: season.charAt(0).toUpperCase() + season.slice(1),
    style: style.charAt(0).toUpperCase() + style.slice(1),
    items: [],
    color_palette: [],
    tips: []
  };

  // Select colors based on season
  if (["spring", "summer"].includes(season.toLowerCase())) {
    outfit.color_palette = [...COLOR_HARMONY.pastel, ...COLOR_HARMONY.warm.slice(0, 3)];
  } else if (["fall", "winter"].includes(season.toLowerCase())) {
    outfit.color_palette = [...COLOR_HARMONY.warm.slice(0, 3), "black", "gray", "navy"];
  } else {
    outfit.color_palette = COLOR_HARMONY.neutral;
  }

  // Select items for each category
  for (const [category, items] of Object.entries(baseItems)) {
    if (items && items.length > 0) {
      outfit.items.push({
        category: category.charAt(0).toUpperCase() + category.slice(1),
        suggestion: items[0].charAt(0).toUpperCase() + items[0].slice(1)
      });
    }
  }

  // Add styling tips based on preference
  const stylePref = style.toLowerCase();
  if (stylePref === "modern") {
    outfit.tips = ["Stick to clean lines and minimal accessories", "Choose solid colors over patterns", "Opt for tailored fits"];
  } else if (stylePref === "classic") {
    outfit.tips = ["Choose timeless pieces that never go out of style", "Stick to neutral colors", "Quality over quantity"];
  } else if (stylePref === "edgy") {
    outfit.tips = ["Add statement pieces like leather jackets", "Mix textures (leather + denim)", "Don't be afraid to experiment"];
  } else if (stylePref === "bohemian") {
    outfit.tips = ["Layer jewelry and accessories", "Mix patterns and textures", "Flowy fabrics work best"];
  } else {
    outfit.tips = ["Dress for comfort and confidence", "Accessorize appropriately", "Pay attention to fit"];
  }

  res.status(200).setHeader("Access-Control-Allow-Origin", "*").json(outfit);
}
