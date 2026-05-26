// ============================================================
// DESIGN TOKENS — FAMiLiEN TiSCH
// Dark mode, terra cotta accent
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

  // Amber (kept for backward compat)
  amber:     "#c8993a",
  amberDim:  "#8a6a28",
  amberBg:   "#1a1508",

  // Terra Cotta — FAMiLiEN TiSCH Hauptfarbe
  terracotta:    "#c4622d",
  terracottaDim: "#8a3d1e",
  terracottaBg:  "#170a06",

  // Warmes Herzrot
  heart:     "#b86878",
  heartDim:  "#7a4050",
  heartBg:   "#140a0d",

  // Kühles Kopfblau
  red:       "#c0574a",
  redDim:    "#8a3d34",
  redBg:     "#160d0b",
  blue:      "#5a8fb8",
  blueDim:   "#3d6480",
  blueBg:    "#0a1118",
};

export const typography = {
  display: "Georgia",
  body:    "System",

  hero:   36,
  h1:     28,
  h2:     20,
  h3:     16,
  body:   15,
  small:  13,
  xs:     11,

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

export const agentColors = {
  // Epistemische Agenten
  "Systemisch":           { accent: "#6b9e6e", bg: "#0a120a" },
  "Tiefenpsychologisch":  { accent: "#9e7b6e", bg: "#120e0a" },
  "Empirisch-Rational":   { accent: "#6e8e9e", bg: "#0a0f12" },
  "Philosophisch":        { accent: "#9e9a6e", bg: "#12110a" },
  "Ethisch":              { accent: "#8e7eb8", bg: "#0e0b14" },
  "Abwägung":             { accent: "#b8a46e", bg: "#141008" },
  "Strategisch":          { accent: "#7e9eb8", bg: "#0a1018" },
  "Risiko":               { accent: "#c07050", bg: "#160a08" },

  // Familien-Agenten
  "Familiensystemisch":   { accent: "#c4622d", bg: "#170a06" },
  "Therapeutisch":        { accent: "#b86878", bg: "#140a0d" },
  "Biografisch":          { accent: "#9a8870", bg: "#120e0a" },
  "Pädagogisch":          { accent: "#7aaa7a", bg: "#0a140a" },
  "Achtsam":              { accent: "#8abab8", bg: "#0a1414" },
  "Aus Kinderaugen":      { accent: "#d4a050", bg: "#1a1006" },
  "Narrativ":             { accent: "#9878b8", bg: "#100c14" },
  "Körperorientiert":     { accent: "#b89870", bg: "#140e08" },
  "Psychologisch":        { accent: "#a88ab8", bg: "#120c14" },
  "Soziologisch":         { accent: "#78a898", bg: "#0a1210" },
  "Kulturell":            { accent: "#c8986e", bg: "#1a0e08" },
  "Spirituell":           { accent: "#a0a8c8", bg: "#0c0e14" },
  "Phänomenologisch":     { accent: "#8898b8", bg: "#0a0e14" },
  "Neurodivergent":       { accent: "#98c898", bg: "#0a140a" },
};
