import React, { useState, useRef } from "react";
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, KeyboardAvoidingView, Platform, Animated,
  ActivityIndicator, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors, spacing, radius, typography } from "../theme";
import { askFamilyTable, LOADING_LABELS } from "../services/api";

const CATEGORIES = [
  { id: "all",          label: "Alle Themen" },
  { id: "erziehung",    label: "Erziehung" },
  { id: "beziehung",    label: "Beziehung" },
  { id: "generationen", label: "Generationen" },
  { id: "krise",        label: "Krise" },
];

const EXAMPLES = {
  all: [
    "Mein Kind ist 14 und ich komme kaum noch an es heran. Was steckt dahinter?",
    "Wir streiten als Paar ständig über dieselben Themen — ohne Fortschritt.",
    "Meine Eltern erwarten Dinge von mir, die ich nicht mehr geben kann.",
    "Wie viel Selbständigkeit braucht ein 10-jähriges Kind wirklich?",
  ],
  erziehung: [
    "Mein Kind lügt mich an — Konsequenzen helfen nicht. Was mache ich?",
    "Mein Partner und ich erziehen grundverschieden. Schadet das dem Kind?",
    "Wie setze ich Grenzen, ohne die Beziehung zu meinem Kind zu beschädigen?",
    "Mein Kind ist hochsensibel — ich weiß nicht wie ich es richtig begleite.",
  ],
  beziehung: [
    "Wir lieben uns — aber reden kaum noch miteinander. Ist das noch normal?",
    "Soll ich meinem Partner sagen, was ich wirklich denke, auch wenn es wehtut?",
    "Warum kämpfe ich in meiner Familie ständig um Anerkennung?",
    "Ich fühle mich in meiner Ehe allein. Was fehlt mir wirklich?",
  ],
  generationen: [
    "Meine Eltern mischen sich in meine Erziehung ein. Wie ziehe ich Grenzen?",
    "Oma und Opa sagen dem Kind Dinge, die ich ablehne. Was tun?",
    "Ich pflege meine Eltern — und verliere dabei mich selbst.",
    "Zwischen meinen Eltern und meinem Partner gibt es dauerhaft Spannung.",
  ],
  krise: [
    "Unsere Familie steckt in einer Dauerkrise. Wo fange ich an?",
    "Mein Kind will keinen Kontakt mehr zu mir. Was kann ich noch tun?",
    "Trennung oder bleiben? Ich weiß es nicht mehr.",
    "Nach der Trennung kämpfen wir um die Kinder — und sie leiden darunter.",
  ],
};

function FamilienLogo() {
  return (
    <View style={styles.logoMark}>
      <View style={styles.logoTabletop} />
      <View style={styles.logoLeg} />
      <View style={styles.logoBase} />
      <View style={styles.logoDots}>
        {[colors.terracotta, colors.heart, "#9a8870", "#7aaa7a"].map((c, i) => (
          <View key={i} style={[styles.logoDot, { backgroundColor: c }]} />
        ))}
      </View>
    </View>
  );
}

export default function HomeScreen({ navigation }) {
  const [question, setQuestion]   = useState("");
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [category, setCategory]   = useState("all");
  const inputRef  = useRef(null);
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1, duration: 800, useNativeDriver: true,
    }).start();
  }, []);

  const submit = async () => {
    const q = question.trim();
    if (!q || q.length < 5 || loading) return;
    setError(null);
    setLoading(true);
    try {
      const result = await askFamilyTable(q, category);
      navigation.navigate("Result", { question: q, result });
    } catch (e) {
      setError(e.message || "Verbindungsfehler. Prüfe deine Internetverbindung.");
    } finally {
      setLoading(false);
    }
  };

  const useExample = (text) => {
    setQuestion(text);
    inputRef.current?.focus();
  };

  const examples      = EXAMPLES[category]      || EXAMPLES.all;
  const loadingAgents = LOADING_LABELS[category] || LOADING_LABELS.all;

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <StatusBar barStyle="light-content" backgroundColor={colors.bg} />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {/* Hero */}
          <Animated.View style={[styles.hero, { opacity: fadeAnim }]}>
            <FamilienLogo />
            <Text style={styles.eyebrow}>FAMILIÄRE PERSPEKTIVEN AM TiSCH</Text>
            <Text style={styles.title}>FAMiLiEN TiSCH</Text>
            <Text style={styles.subtitle}>
              Vier Perspektiven denken gleichzeitig — systemisch, therapeutisch,
              biografisch. Kein Ratschlag, keine Diagnose. Ehrliche Orientierung.
            </Text>
          </Animated.View>

          {/* Kategorie-Filter */}
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.filterRow}
            contentContainerStyle={styles.filterContent}
          >
            {CATEGORIES.map((cat) => (
              <TouchableOpacity
                key={cat.id}
                style={[styles.filterChip, category === cat.id && styles.filterChipActive]}
                onPress={() => setCategory(cat.id)}
                activeOpacity={0.7}
              >
                <Text style={[
                  styles.filterChipText,
                  category === cat.id && styles.filterChipTextActive,
                ]}>
                  {cat.label}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>

          {/* Eingabe */}
          <View style={styles.inputCard}>
            <TextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Welche familiäre Situation legen wir auf den Tisch?"
              placeholderTextColor={colors.textMuted}
              value={question}
              onChangeText={setQuestion}
              multiline
              maxLength={600}
              onSubmitEditing={submit}
              editable={!loading}
              returnKeyType="send"
              selectionColor={colors.terracotta}
            />
            <TouchableOpacity
              style={[
                styles.submitBtn,
                (!question.trim() || loading) && styles.submitBtnDisabled,
              ]}
              onPress={submit}
              disabled={!question.trim() || loading}
              activeOpacity={0.8}
            >
              {loading ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Text style={styles.submitBtnText}>Am Tisch besprechen</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Lade-Zustand */}
          {loading && (
            <View style={styles.loadingCard}>
              <View style={styles.loadingAgents}>
                {loadingAgents.map((label, i) => (
                  <View key={i} style={styles.loadingAgent}>
                    <ActivityIndicator size="small" color={colors.terracotta} />
                    <Text style={styles.loadingAgentText}>{label}</Text>
                  </View>
                ))}
              </View>
              <Text style={styles.loadingHint}>Vier Perspektiven denken für euch…</Text>
            </View>
          )}

          {/* Fehler */}
          {error && (
            <View style={styles.errorCard}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {/* Beispielfragen */}
          {!loading && (
            <View style={styles.examples}>
              <Text style={styles.examplesLabel}>Themen am TiSCH</Text>
              {examples.map((ex, i) => (
                <TouchableOpacity
                  key={`${category}-${i}`}
                  style={styles.exampleChip}
                  onPress={() => useExample(ex)}
                  activeOpacity={0.7}
                >
                  <Text style={styles.exampleText}>{ex}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: colors.bg },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl },

  // Logo
  logoMark:    { alignItems: "center", marginBottom: spacing.md },
  logoTabletop: { width: 52, height: 5, backgroundColor: colors.terracotta, borderRadius: 2, opacity: 0.9 },
  logoLeg:     { width: 5, height: 20, backgroundColor: colors.textMuted, borderRadius: 1, marginTop: 2 },
  logoBase:    { width: 30, height: 3, backgroundColor: colors.textMuted, borderRadius: 1.5, marginTop: 1, opacity: 0.6 },
  logoDots:    { flexDirection: "row", gap: 5, marginTop: spacing.sm },
  logoDot:     { width: 7, height: 7, borderRadius: 3.5 },

  // Hero
  hero: {
    alignItems: "center",
    paddingTop:    spacing.xxl,
    paddingBottom: spacing.lg,
  },
  eyebrow: {
    fontSize:    typography.xs,
    letterSpacing: 2,
    color:       colors.terracotta,
    fontWeight:  "600",
    marginBottom: spacing.sm,
  },
  title: {
    fontSize:    typography.hero,
    color:       colors.textPrimary,
    fontWeight:  "700",
    marginBottom: spacing.md,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize:   typography.small,
    color:      colors.textSecondary,
    textAlign:  "center",
    lineHeight: 22,
    maxWidth:   320,
  },

  // Filter
  filterRow:     { marginBottom: spacing.md },
  filterContent: { paddingRight: spacing.lg, gap: spacing.sm, flexDirection: "row" },
  filterChip: {
    paddingHorizontal: spacing.md,
    paddingVertical:   spacing.xs + 2,
    borderRadius:      radius.full,
    borderWidth:       1,
    borderColor:       colors.border,
    backgroundColor:   colors.bgCard,
  },
  filterChipActive: {
    borderColor:     colors.terracotta,
    backgroundColor: "rgba(196,98,45,0.12)",
  },
  filterChipText:       { fontSize: typography.small, color: colors.textMuted,     fontWeight: "500" },
  filterChipTextActive: { fontSize: typography.small, color: colors.terracotta,    fontWeight: "600" },

  // Input
  inputCard: {
    backgroundColor: colors.bgCard,
    borderRadius:    radius.lg,
    borderWidth:     1,
    borderColor:     colors.border,
    padding:         spacing.md,
    marginBottom:    spacing.lg,
  },
  input: {
    fontSize:        typography.body,
    color:           colors.textPrimary,
    minHeight:       80,
    textAlignVertical: "top",
    lineHeight:      24,
    marginBottom:    spacing.md,
  },
  submitBtn: {
    backgroundColor: colors.terracotta,
    borderRadius:    radius.md,
    paddingVertical: spacing.md,
    alignItems:      "center",
  },
  submitBtnDisabled: {
    backgroundColor: colors.terracottaDim,
    opacity:         0.5,
  },
  submitBtnText: {
    color:       "#fff",
    fontWeight:  "700",
    fontSize:    typography.body,
    letterSpacing: 0.3,
  },

  // Laden
  loadingCard: {
    backgroundColor: colors.bgCard,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.border,
    padding:         spacing.lg,
    marginBottom:    spacing.lg,
    alignItems:      "center",
  },
  loadingAgents: {
    flexDirection: "row",
    flexWrap:      "wrap",
    justifyContent: "center",
    gap:           spacing.sm,
    marginBottom:  spacing.md,
  },
  loadingAgent: {
    flexDirection:  "row",
    alignItems:     "center",
    gap:            spacing.xs,
    backgroundColor: colors.bgInput,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius:   radius.full,
  },
  loadingAgentText: { fontSize: typography.xs, color: colors.textSecondary },
  loadingHint:      { fontSize: typography.small, color: colors.textMuted, fontStyle: "italic" },

  // Fehler
  errorCard: {
    backgroundColor: colors.redBg,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.redDim,
    padding:         spacing.md,
    marginBottom:    spacing.lg,
  },
  errorText: { color: colors.red, fontSize: typography.small, lineHeight: 20 },

  // Beispiele
  examples:      { gap: spacing.sm },
  examplesLabel: {
    fontSize:     typography.xs,
    color:        colors.textMuted,
    letterSpacing: 1.5,
    fontWeight:   "600",
    marginBottom: spacing.xs,
  },
  exampleChip: {
    backgroundColor: colors.bgCard,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.border,
    padding:         spacing.md,
  },
  exampleText: { fontSize: typography.small, color: colors.textSecondary, lineHeight: 20 },
});
