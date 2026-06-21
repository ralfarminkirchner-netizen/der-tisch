// ============================================================
// DESIGN TOKENS — Der Tisch
// Dark mode first, same palette as the web app
// ============================================================

export const colors = {
  // Backgrounds
  bg:        "#0e0d0b",
  bgCard:    "#161410",
  bgInput:   "#1a1815",
  bgHover:   "#201e1a",
  border:    "#2a2720",
  borderSub: "#1f1d19",

  // Text
  textPrimary:   "#f0ede6",
  textSecondary: "#9a9080",
  textMuted:     "#5a5448",

  // Accents
  amber:     "#c8993a",
  amberDim:  "#8a6a28",
  amberBg:   "#1a1508",
  red:       "#c0574a",
  redDim:    "#8a3d34",
  redBg:     "#160d0b",
  blue:      "#5a8fb8",
  blueDim:   "#3d6480",
  blueBg:    "#0a1118",

  // FAMiLiEN TiSCH
  familie:    "#C4622D",
  familieDim: "#8a3d1e",
  familieBg:  "#140a06",
};

export const typography = {
  display: "Georgia", // serif display font (system)
  body:    "System",

  // Sizes
  hero:   36,
  h1:     28,
  h2:     20,
  h3:     16,
  body:   15,
  small:  13,
  xs:     11,

  // Line heights
  lineHeightTight:  1.2,
  lineHeightNormal: 1.5,
  lineHeightLoose:  1.75,
};

export const spacing = {
  xs:  4,
  sm:  8,
  md:  16,
  lg:  24,
  xl:  32,
  xxl: 48,
};

export const radius = {
  sm:   6,
  md:   12,
  lg:   18,
  full: 999,
};

// Agent role colors (generic Der Tisch)
export const agentColors = {
  "Systemisch":         { accent: "#6b9e6e", bg: "#0a120a" },
  "Tiefenpsychologisch":{ accent: "#9e7b6e", bg: "#120e0a" },
  "Empirisch-Rational": { accent: "#6e8e9e", bg: "#0a0f12" },
  "Philosophisch":      { accent: "#9e9a6e", bg: "#12110a" },

  // FAMiLiEN TiSCH perspectives
  "Systemische Therapeutin":    { accent: "#C4622D", bg: "#140a06" },
  "Bindungsexpertin":           { accent: "#D4845A", bg: "#160b07" },
  "Erziehungsberaterin":        { accent: "#B85C2A", bg: "#130903" },
  "Paartherapeutin":            { accent: "#A84E20", bg: "#120803" },
  "Traumaexpertin":             { accent: "#C87050", bg: "#150905" },
  "Kind-Perspektive":           { accent: "#E0956A", bg: "#170c06" },
  "Großeltern & Generationen":  { accent: "#9A5A3A", bg: "#110706" },
  "Konfliktmediation":          { accent: "#CC7040", bg: "#150a05" },
  "Kommunikationsexpertin":     { accent: "#B86840", bg: "#140905" },
  "Identität & Rolle":          { accent: "#D07850", bg: "#160b06" },
};
