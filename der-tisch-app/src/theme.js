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

// Agent role colors — all 19 German + 10 English agent roles
export const agentColors = {
  "Systemisch":           { accent: "#6b9e6e", bg: "#0a120a" },
  "Tiefenpsychologisch":  { accent: "#9e7b6e", bg: "#120e0a" },
  "Empirisch-Rational":   { accent: "#6e8e9e", bg: "#0a0f12" },
  "Philosophisch":        { accent: "#9e9a6e", bg: "#12110a" },
  "Ethisch":              { accent: "#8e7a9e", bg: "#0f0a12" },
  "Abwägung":             { accent: "#c8993a", bg: "#120e03" },
  "Strategisch":          { accent: "#6e9e8e", bg: "#0a120f" },
  "Risiko":               { accent: "#c07060", bg: "#120a0a" },
  "Pädagogisch":          { accent: "#8eb06e", bg: "#0f120a" },
  "Neurodivergent":       { accent: "#7e8eb0", bg: "#0a0f14" },
  "Aus Kinderaugen":      { accent: "#d4845a", bg: "#130d08" },
  "Therapeutisch":        { accent: "#9e8a6e", bg: "#12100a" },
  "Psychologisch":        { accent: "#a07e9e", bg: "#110a12" },
  "Achtsam":              { accent: "#8ea89e", bg: "#0a120f" },
  "Familiensystemisch":   { accent: "#c4622d", bg: "#130a06" },
  "Biografisch":          { accent: "#9e8070", bg: "#120f0d" },
  "Soziologisch":         { accent: "#7890a8", bg: "#0a0f13" },
  "Kulturell":            { accent: "#a89060", bg: "#13110a" },
  "Spirituell":           { accent: "#a088b8", bg: "#0f0a14" },
  "Narrativ":             { accent: "#b88068", bg: "#130e0a" },
  "Körperorientiert":     { accent: "#7e9e7e", bg: "#0a120a" },
  "Phänomenologisch":     { accent: "#8890a8", bg: "#0a0f12" },
  // English roles
  "Systemic":             { accent: "#6b9e6e", bg: "#0a120a" },
  "Depth-Psychological":  { accent: "#9e7b6e", bg: "#120e0a" },
  "Empirical-Rational":   { accent: "#6e8e9e", bg: "#0a0f12" },
  "Philosophical":        { accent: "#9e9a6e", bg: "#12110a" },
  "Ethical":              { accent: "#8e7a9e", bg: "#0f0a12" },
  "Weighing":             { accent: "#c8993a", bg: "#120e03" },
  "Strategic":            { accent: "#6e9e8e", bg: "#0a120f" },
  "Risk":                 { accent: "#c07060", bg: "#120a0a" },
  "Neurodivergent":       { accent: "#7e8eb0", bg: "#0a0f14" },
  "Through Children's Eyes": { accent: "#d4845a", bg: "#130d08" },
};
