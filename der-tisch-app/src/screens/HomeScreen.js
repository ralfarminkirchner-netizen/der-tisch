import React, { useState, useRef } from "react";
import {
  View, Text, TextInput, TouchableOpacity, ScrollView,
  StyleSheet, KeyboardAvoidingView, Platform, Animated,
  ActivityIndicator, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors, spacing, radius, typography } from "../theme";
import { askFamilientisch } from "../services/api";

const PERSPEKTIVEN = [
  { id: 'systemisch',    de: 'Systemische Therapeutin',  color: '#C4622D' },
  { id: 'bindung',       de: 'Bindungsexpertin',          color: '#D4845A' },
  { id: 'erziehung',     de: 'Erziehungsberaterin',       color: '#B85C2A' },
  { id: 'paar',          de: 'Paartherapeutin',           color: '#A84E20' },
  { id: 'trauma',        de: 'Traumaexpertin',            color: '#C87050' },
  { id: 'kind',          de: 'Kind-Perspektive',          color: '#E0956A' },
  { id: 'grosseltern',   de: 'Generationen',              color: '#9A5A3A' },
  { id: 'konflikt',      de: 'Konfliktmediation',         color: '#CC7040' },
  { id: 'kommunikation', de: 'Kommunikation',             color: '#B86840' },
  { id: 'identitaet',    de: 'Identität & Rolle',         color: '#D07850' },
];

const EXAMPLES = [
  'Mein Kind macht, was es will — wie setze ich Grenzen?',
  'Teenager-Phase: Nähe und Autonomie balancieren.',
  'Meine Mutter mischt sich ständig ein.',
  'Geschwisterkonflikt — wie vermitteln?',
  'Welche Muster aus meiner Herkunftsfamilie trage ich weiter?',
  'Wie spreche ich Trennung mit Kindern an?',
];

const TONES = [
  { id: 'achtsam',     label: 'Achtsam' },
  { id: 'direkt',      label: 'Direkt' },
  { id: 'ressourcen',  label: 'Ressourcen' },
];

export default function HomeScreen({ navigation }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activePerspectives, setActivePerspectives] = useState(
    new Set(PERSPEKTIVEN.map(p => p.id))
  );
  const [tone, setTone] = useState('achtsam');
  const inputRef = useRef(null);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1, duration: 800, useNativeDriver: true,
    }).start();
  }, []);

  const togglePerspective = (id) => {
    setActivePerspectives(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleAll = () => {
    if (activePerspectives.size === PERSPEKTIVEN.length) {
      setActivePerspectives(new Set([PERSPEKTIVEN[0].id]));
    } else {
      setActivePerspectives(new Set(PERSPEKTIVEN.map(p => p.id)));
    }
  };

  const submit = async () => {
    const q = question.trim();
    if (!q || q.length < 5 || loading) return;
    setError(null);
    setLoading(true);
    try {
      const selectedPersp = PERSPEKTIVEN.filter(p => activePerspectives.has(p.id));
      const result = await askFamilientisch(q, selectedPersp, tone, 'de');
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
            <View style={styles.logoWrap}>
              <View style={styles.logoBar} />
              <View style={styles.logoLeg} />
              <View style={styles.logoBase} />
            </View>
            <Text style={styles.eyebrow}>FAMILIÄRE PERSPEKTIVEN AM TiSCH</Text>
            <Text style={styles.title}>
              FAM<Text style={styles.titleAccent}>i</Text>L<Text style={styles.titleAccent}>i</Text>EN{"\n"}T<Text style={styles.titleAccent}>i</Text>SCH
            </Text>
            <Text style={styles.subtitle}>
              Multiperspektivische KI-Analyse für das Familienleben
            </Text>
          </Animated.View>

          {/* Input */}
          <View style={styles.inputCard}>
            <TextInput
              ref={inputRef}
              style={styles.input}
              placeholder="Beschreiben Sie Ihre familiäre Situation..."
              placeholderTextColor={colors.textMuted}
              value={question}
              onChangeText={setQuestion}
              multiline
              maxLength={600}
              editable={!loading}
              selectionColor={colors.familie}
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
                <Text style={styles.submitBtnText}>An den TiSCH</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Loading */}
          {loading && (
            <View style={styles.loadingCard}>
              <Text style={styles.loadingHint}>
                {activePerspectives.size} Perspektiven denken parallel…
              </Text>
              <View style={styles.loadingAgents}>
                {Array.from(activePerspectives).slice(0, 4).map(id => {
                  const p = PERSPEKTIVEN.find(x => x.id === id);
                  return p ? (
                    <View key={p.id} style={styles.loadingAgent}>
                      <ActivityIndicator size="small" color={p.color} />
                      <Text style={styles.loadingAgentText}>
                        {p.de.split(' ')[0]}
                      </Text>
                    </View>
                  ) : null;
                })}
              </View>
            </View>
          )}

          {/* Error */}
          {error && (
            <View style={styles.errorCard}>
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {/* Perspective selector */}
          <View style={styles.section}>
            <View style={styles.sectionRow}>
              <Text style={styles.sectionLabel}>Perspektiven</Text>
              <TouchableOpacity onPress={toggleAll} activeOpacity={0.7}>
                <Text style={styles.toggleAllBtn}>
                  {activePerspectives.size === PERSPEKTIVEN.length ? 'Keine' : 'Alle'}
                </Text>
              </TouchableOpacity>
            </View>
            <View style={styles.perspGrid}>
              {PERSPEKTIVEN.map(p => {
                const active = activePerspectives.has(p.id);
                return (
                  <TouchableOpacity
                    key={p.id}
                    style={[
                      styles.perspChip,
                      active && { borderColor: p.color, backgroundColor: p.color + '1A' },
                    ]}
                    onPress={() => togglePerspective(p.id)}
                    activeOpacity={0.7}
                  >
                    <View style={[
                      styles.perspDot,
                      { backgroundColor: active ? p.color : colors.border },
                    ]} />
                    <Text style={[
                      styles.perspLabel,
                      active && { color: colors.textPrimary },
                    ]}>
                      {p.de}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* Tone selector */}
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>Tonalität</Text>
            <View style={styles.toneRow}>
              {TONES.map(t => (
                <TouchableOpacity
                  key={t.id}
                  style={[styles.toneBtn, tone === t.id && styles.toneBtnActive]}
                  onPress={() => setTone(t.id)}
                  activeOpacity={0.7}
                >
                  <Text style={[
                    styles.toneBtnText,
                    tone === t.id && styles.toneBtnTextActive,
                  ]}>
                    {t.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Example questions */}
          {!loading && (
            <View style={styles.examples}>
              <Text style={styles.examplesLabel}>Themen am TiSCH</Text>
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

          {/* KiNTEGRiTY disclaimer */}
          <View style={styles.kiCard}>
            <Text style={styles.kiText}>
              kiNTEGRiTY — Keine Diagnose · Kein Therapie-Ersatz · Vertraulich · Multiperspektivisch
            </Text>
          </View>

        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  scroll: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl,
  },

  // Hero
  hero: {
    alignItems: "center",
    paddingTop: spacing.xl,
    paddingBottom: spacing.lg,
  },
  logoWrap: {
    alignItems: "center",
    marginBottom: spacing.md,
  },
  logoBar: {
    width: 44,
    height: 5,
    backgroundColor: colors.familie,
    borderRadius: 2,
  },
  logoLeg: {
    width: 4,
    height: 14,
    backgroundColor: colors.familieDim,
    borderRadius: 2,
    marginTop: 1,
  },
  logoBase: {
    width: 28,
    height: 4,
    backgroundColor: colors.familieDim,
    borderRadius: 2,
    marginTop: 1,
  },
  eyebrow: {
    fontSize: typography.xs,
    letterSpacing: 2,
    color: colors.textMuted,
    fontWeight: "600",
    marginBottom: spacing.sm,
    textAlign: "center",
  },
  title: {
    fontSize: 38,
    color: colors.textPrimary,
    fontWeight: "300",
    textAlign: "center",
    letterSpacing: -1,
    lineHeight: 44,
    marginBottom: spacing.md,
  },
  titleAccent: {
    color: colors.familie,
  },
  subtitle: {
    fontSize: typography.small,
    color: colors.textSecondary,
    textAlign: "center",
    lineHeight: 22,
    maxWidth: 300,
  },

  // Input
  inputCard: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  input: {
    fontSize: typography.body,
    color: colors.textPrimary,
    minHeight: 80,
    textAlignVertical: "top",
    lineHeight: 24,
    marginBottom: spacing.md,
  },
  submitBtn: {
    backgroundColor: colors.familie,
    borderRadius: radius.md,
    paddingVertical: spacing.md,
    alignItems: "center",
  },
  submitBtnDisabled: {
    backgroundColor: colors.familieDim,
    opacity: 0.5,
  },
  submitBtnText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: typography.body,
    letterSpacing: 0.3,
  },

  // Loading
  loadingCard: {
    backgroundColor: colors.familieBg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.familieDim,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    alignItems: "center",
  },
  loadingHint: {
    fontSize: typography.small,
    color: colors.textSecondary,
    fontStyle: "italic",
    marginBottom: spacing.md,
  },
  loadingAgents: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.sm,
  },
  loadingAgent: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    backgroundColor: colors.bgInput,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius.full,
  },
  loadingAgentText: {
    fontSize: typography.xs,
    color: colors.textSecondary,
  },

  // Error
  errorCard: {
    backgroundColor: colors.redBg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.redDim,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  errorText: {
    color: colors.red,
    fontSize: typography.small,
    lineHeight: 20,
  },

  // Sections
  section: {
    marginBottom: spacing.lg,
  },
  sectionRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.sm,
  },
  sectionLabel: {
    fontSize: typography.xs,
    color: colors.textMuted,
    letterSpacing: 1.5,
    fontWeight: "600",
    marginBottom: spacing.sm,
  },
  toggleAllBtn: {
    fontSize: typography.xs,
    color: colors.familie,
    fontWeight: "600",
    letterSpacing: 0.5,
  },

  // Perspective grid
  perspGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.xs,
  },
  perspChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    borderRadius: radius.full,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.bgCard,
  },
  perspDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  perspLabel: {
    fontSize: typography.xs,
    color: colors.textMuted,
    fontWeight: "500",
  },

  // Tone selector
  toneRow: {
    flexDirection: "row",
    gap: spacing.xs,
  },
  toneBtn: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: "center",
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.bgCard,
  },
  toneBtnActive: {
    borderColor: colors.familie,
    backgroundColor: colors.familieBg,
  },
  toneBtnText: {
    fontSize: typography.xs,
    color: colors.textMuted,
    fontWeight: "500",
    letterSpacing: 0.3,
  },
  toneBtnTextActive: {
    color: colors.familie,
    fontWeight: "700",
  },

  // Examples
  examples: {
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  examplesLabel: {
    fontSize: typography.xs,
    color: colors.textMuted,
    letterSpacing: 1.5,
    fontWeight: "600",
    marginBottom: spacing.xs,
  },
  exampleChip: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  exampleText: {
    fontSize: typography.small,
    color: colors.textSecondary,
    lineHeight: 20,
  },

  // KiNTEGRiTY
  kiCard: {
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingTop: spacing.md,
    marginTop: spacing.sm,
  },
  kiText: {
    fontSize: typography.xs,
    color: colors.textMuted,
    textAlign: "center",
    letterSpacing: 0.3,
    lineHeight: 18,
  },
});
