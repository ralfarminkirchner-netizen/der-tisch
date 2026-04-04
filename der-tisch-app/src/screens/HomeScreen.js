import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, KeyboardAvoidingView, Platform, Animated,
  ActivityIndicator, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors, spacing, radius, typography } from "../theme";
import { askTheTable } from "../services/api";

const EXAMPLES = [
  "Ich liebe meine Partnerin, aber unsere intellektuellen Welten haben sich entfremdet. Ist das ein Trennungsgrund?",
  "Soll ich den sicheren Job behalten, obwohl er mich innerlich abstumpft?",
  "Warum kann ich mich nicht entscheiden, obwohl ich alle Fakten kenne?",
  "Ich helfe immer allen — aber fühle mich dabei oft ausgenutzt. Was steckt dahinter?",
  "Klimawandel — was kann ich persönlich tun?",
  "Ist KI eine Bedrohung oder eine Chance?",
];

// Agent accent colors — matching web app
const AGENT_COLORS = {
  "Philosophisch":      "#7B52D4",
  "Tiefenpsychologisch":"#5C3D9E",
  "Systemisch":         "#2E8B57",
  "Empirisch-Rational": "#2980B9",
  "Therapeutisch":      "#3498DB",
  "Achtsam":            "#17A89B",
  "Ethisch":            "#D35400",
  "Strategisch":        "#1E7A8C",
  "Pädagogisch":        "#E67E22",
  "Juridisch":          "#922B21",
  "Spirituell":         "#8E44AD",
  "Narrativ":           "#2471A3",
  "Neurodivergent":     "#A8D530",
  "Ökonomisch":         "#27AE60",
  "Biografisch":        "#B7950B",
  "Kulturell":          "#7A9E50",
};

function getAgentColor(role) {
  if (AGENT_COLORS[role]) return AGENT_COLORS[role];
  for (const [key, color] of Object.entries(AGENT_COLORS)) {
    if (role && role.includes(key)) return color;
  }
  return colors.amber;
}

// TiSCH Logo mark
function TableLogo() {
  return (
    <View style={styles.logoMark}>
      <View style={styles.logoTabletop} />
      <View style={styles.logoLegs}>
        <View style={styles.logoLeg} />
        <View style={styles.logoLeg} />
      </View>
      <View style={styles.logoDots}>
        {[colors.amber, "#6b9e6e", "#9e7b6e", "#6e8e9e"].map((c, i) => (
          <View key={i} style={[styles.logoDot, { backgroundColor: c }]} />
        ))}
      </View>
    </View>
  );
}

// KI-Auto Expert Preview card
function ExpertPreviewCard({ experts, loading }) {
  if (!loading && (!experts || experts.length === 0)) return null;

  return (
    <View style={styles.expertPreviewCard}>
      <Text style={styles.expertPreviewLabel}>
        {loading ? "KI wählt Perspektiven…" : "Diese Stimmen kommen an den Tisch"}
      </Text>
      {loading ? (
        <View style={styles.expertPreviewLoading}>
          <ActivityIndicator size="small" color={colors.amber} />
          <Text style={styles.expertPreviewLoadingText}>Analysiere Frage…</Text>
        </View>
      ) : (
        <View style={styles.expertChips}>
          {experts.map((e, i) => {
            const color = getAgentColor(e.role);
            return (
              <View key={i} style={[styles.expertChip, { borderColor: color + "55" }]}>
                <View style={[styles.expertChipDot, { backgroundColor: color }]} />
                <Text style={[styles.expertChipText, { color }]}>{e.role}</Text>
              </View>
            );
          })}
        </View>
      )}
    </View>
  );
}

export default function HomeScreen({ navigation }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);

  // KI-Auto expert preview state
  const [previewExperts, setPreviewExperts]       = useState([]);
  const [previewLoading, setPreviewLoading]       = useState(false);
  const [previewQuestion, setPreviewQuestion]     = useState("");

  const inputRef    = useRef(null);
  const fadeAnim    = useRef(new Animated.Value(0)).current;
  const previewTimer= useRef(null);

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1, duration: 800, useNativeDriver: true,
    }).start();
  }, []);

  // Debounced expert preview: fires 1.2s after user stops typing
  useEffect(() => {
    if (previewTimer.current) clearTimeout(previewTimer.current);

    const q = question.trim();
    if (!q || q.length < 12) {
      setPreviewExperts([]);
      setPreviewLoading(false);
      setPreviewQuestion("");
      return;
    }
    if (q === previewQuestion) return; // already previewed this question

    setPreviewLoading(true);

    previewTimer.current = setTimeout(async () => {
      try {
        // Use ask-simple to get a quick preview of which agents would be chosen
        const res = await fetch("https://der-tisch-production.up.railway.app/api/ask-simple", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q, lang: "de", stil: "alltag", tone: "achtsam", register: "" }),
        });
        if (!res.ok) throw new Error();
        const data = await res.json();
        const experts = (data.perspectives || []).map((p) => ({
          role: (p.rolle || p.role || "Experte").replace(/^\[(custom|method)\]/, "").trim(),
          snippet: (p.kernanalyse || "").slice(0, 80),
        }));
        setPreviewExperts(experts);
        setPreviewQuestion(q);
      } catch (e) {
        setPreviewExperts([]);
      } finally {
        setPreviewLoading(false);
      }
    }, 1200);

    return () => clearTimeout(previewTimer.current);
  }, [question]);

  const submit = async () => {
    const q = question.trim();
    if (!q || q.length < 5 || loading) return;
    setError(null);
    setLoading(true);
    try {
      const result = await askTheTable(q);
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

  const hasPreview = previewExperts.length > 0 || previewLoading;

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
          <Animated.View style={[styles.hero, { opacity: fadeAnim }]}>
            <TableLogo />
            <Text style={styles.eyebrow}>EPISTEMISCHER PERSPEKTIVENRAUM</Text>
            <Text style={styles.title}>Der Tisch</Text>
            <Text style={styles.subtitle}>
              Keine absolute Wahrheit. Mehrere Methoden, die gleichzeitig denken —
              gefolgt von schonungsloser Reibung und ehrlicher Integration.
            </Text>
          </Animated.View>

          {/* Input */}
          <View style={styles.inputCard}>
            <TextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Welche Frage legen wir auf den Tisch?"
              placeholderTextColor={colors.textMuted}
              value={question}
              onChangeText={setQuestion}
              multiline
              maxLength={600}
              onSubmitEditing={submit}
              editable={!loading}
              returnKeyType="send"
              selectionColor={colors.amber}
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
                <ActivityIndicator size="small" color={colors.bg} />
              ) : (
                <Text style={styles.submitBtnText}>Am Tisch fragen</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* KI-Auto Expert Preview */}
          {!loading && hasPreview && (
            <ExpertPreviewCard experts={previewExperts} loading={previewLoading} />
          )}

          {/* Loading state */}
          {loading && (
            <View style={styles.loadingCard}>
              <View style={styles.loadingDots}>
                {(previewExperts.length > 0 ? previewExperts.map(e => e.role) : ["Systemisch", "Tiefenpsychologisch", "Empirisch-Rational", "Philosophisch"]).map((label, i) => (
                  <View key={i} style={styles.loadingAgent}>
                    <ActivityIndicator size="small" color={getAgentColor(label)} />
                    <Text style={styles.loadingAgentText}>{label}</Text>
                  </View>
                ))}
              </View>
              <Text style={styles.loadingHint}>Perspektiven werden erarbeitet…</Text>
            </View>
          )}

          {/* Error */}
          {error && (
            <View style={styles.errorCard}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {/* Example questions */}
          {!loading && (
            <View style={styles.examples}>
              <Text style={styles.examplesLabel}>Beispielfragen</Text>
              {EXAMPLES.map((ex, i) => (
                <TouchableOpacity
                  key={i}
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
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl },

  // Logo
  logoMark:    { alignItems: "center", marginBottom: spacing.md },
  logoTabletop:{ width: 36, height: 4, backgroundColor: colors.textSecondary, borderRadius: 2 },
  logoLegs:    { flexDirection: "row", justifyContent: "space-between", width: 28, marginTop: 2 },
  logoLeg:     { width: 3, height: 10, backgroundColor: colors.textSecondary, borderRadius: 1 },
  logoDots:    { flexDirection: "row", gap: 4, marginTop: 6 },
  logoDot:     { width: 6, height: 6, borderRadius: 3 },

  // Hero
  hero: { alignItems: "center", paddingTop: spacing.xxl, paddingBottom: spacing.xl },
  eyebrow: { fontSize: typography.xs, letterSpacing: 2, color: colors.amber, fontWeight: "600", marginBottom: spacing.sm },
  title:   { fontSize: typography.hero, color: colors.textPrimary, fontWeight: "700", marginBottom: spacing.md, letterSpacing: -0.5 },
  subtitle:{ fontSize: typography.small, color: colors.textSecondary, textAlign: "center", lineHeight: 22, maxWidth: 320 },

  // Input
  inputCard: {
    backgroundColor: colors.bgCard, borderRadius: radius.lg,
    borderWidth: 1, borderColor: colors.border,
    padding: spacing.md, marginBottom: spacing.md,
  },
  input: {
    fontSize: typography.body, color: colors.textPrimary,
    minHeight: 80, textAlignVertical: "top", lineHeight: 24, marginBottom: spacing.md,
  },
  submitBtn: {
    backgroundColor: colors.amber, borderRadius: radius.md,
    paddingVertical: spacing.md, alignItems: "center",
  },
  submitBtnDisabled: { backgroundColor: colors.amberDim, opacity: 0.5 },
  submitBtnText: { color: colors.bg, fontWeight: "700", fontSize: typography.body, letterSpacing: 0.3 },

  // Expert Preview Card
  expertPreviewCard: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.amber + "30",
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  expertPreviewLabel: {
    fontSize: typography.xs,
    color: colors.amber,
    fontWeight: "700",
    letterSpacing: 1.2,
    textTransform: "uppercase",
    marginBottom: spacing.sm,
  },
  expertPreviewLoading: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  expertPreviewLoadingText: {
    fontSize: typography.small,
    color: colors.textMuted,
    fontStyle: "italic",
  },
  expertChips: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs,
  },
  expertChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    borderWidth: 1,
    borderRadius: radius.full,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    backgroundColor: colors.bgInput,
  },
  expertChipDot: { width: 7, height: 7, borderRadius: 4 },
  expertChipText: { fontSize: typography.xs, fontWeight: "600", letterSpacing: 0.3 },

  // Loading
  loadingCard: {
    backgroundColor: colors.bgCard, borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.border,
    padding: spacing.lg, marginBottom: spacing.lg, alignItems: "center",
  },
  loadingDots: {
    flexDirection: "row", flexWrap: "wrap",
    justifyContent: "center", gap: spacing.sm, marginBottom: spacing.md,
  },
  loadingAgent: {
    flexDirection: "row", alignItems: "center", gap: spacing.xs,
    backgroundColor: colors.bgInput,
    paddingHorizontal: spacing.sm, paddingVertical: spacing.xs,
    borderRadius: radius.full,
  },
  loadingAgentText: { fontSize: typography.xs, color: colors.textSecondary },
  loadingHint: { fontSize: typography.small, color: colors.textMuted, fontStyle: "italic" },

  // Error
  errorCard: {
    backgroundColor: colors.redBg, borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.redDim,
    padding: spacing.md, marginBottom: spacing.lg,
  },
  errorText: { color: colors.red, fontSize: typography.small, lineHeight: 20 },

  // Examples
  examples:      { gap: spacing.sm },
  examplesLabel: { fontSize: typography.xs, color: colors.textMuted, letterSpacing: 1.5, fontWeight: "600", marginBottom: spacing.xs },
  exampleChip:   { backgroundColor: colors.bgCard, borderRadius: radius.md, borderWidth: 1, borderColor: colors.border, padding: spacing.md },
  exampleText:   { fontSize: typography.small, color: colors.textSecondary, lineHeight: 20 },
});
