import React, { useRef, useEffect } from "react";
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  Animated, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors, spacing, radius, typography, agentColors } from "../theme";

// Entfernt das "[method]" / "[custom]" Präfix das das Backend einfügt
function cleanRole(rolle) {
  return rolle.replace(/^\[(method|custom)\]\s*/i, "").trim();
}

// ─── Kleine Bausteine ─────────────────────────────────────

function SectionHeader({ phase, title, subtitle, color }) {
  return (
    <View style={styles.sectionHeader}>
      <Text style={[styles.phaseLabel, { color }]}>{phase}</Text>
      <Text style={styles.sectionTitle}>{title}</Text>
      {subtitle && <Text style={styles.sectionSubtitle}>{subtitle}</Text>}
    </View>
  );
}

function FieldBlock({ label, value, color }) {
  return (
    <View style={styles.fieldBlock}>
      <Text style={[styles.fieldLabel, { color: color || colors.textMuted }]}>{label}</Text>
      <Text style={styles.fieldValue}>{value}</Text>
    </View>
  );
}

function BulletList({ items, color }) {
  if (!items || items.length === 0) return null;
  return (
    <View style={styles.bulletList}>
      {items.map((item, i) => (
        <View key={i} style={styles.bulletRow}>
          <View style={[styles.bullet, { backgroundColor: color }]} />
          <Text style={styles.bulletText}>{item}</Text>
        </View>
      ))}
    </View>
  );
}

// ─── Perspektiven-Karte ───────────────────────────────────

function PerspectiveCard({ perspective, index }) {
  const role = cleanRole(perspective.rolle);
  const ac   = agentColors[role] || { accent: colors.terracotta, bg: colors.bgCard };
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1, duration: 400, delay: index * 120, useNativeDriver: true,
    }).start();
  }, []);

  return (
    <Animated.View style={[styles.perspectiveCard, { backgroundColor: ac.bg, opacity: fadeAnim }]}>
      <View style={[styles.perspectiveAccentBar, { backgroundColor: ac.accent }]} />
      <View style={styles.perspectiveContent}>
        <Text style={[styles.perspectiveRole, { color: ac.accent }]}>
          {role.toUpperCase()}
        </Text>
        <FieldBlock label="Kernanalyse"    value={perspective.kernanalyse}    color={ac.accent} />
        <FieldBlock label="Evidenz"        value={perspective.evidenz}        color={ac.accent} />
        <FieldBlock label="Blinder Fleck"  value={perspective.blinder_fleck}  color={ac.accent} />
      </View>
    </Animated.View>
  );
}

// ─── Reibungs-Karte ───────────────────────────────────────
// Feldnamen: echte_widersprueche, uebersetzungsfehler, uebersehenes

function FrictionCard({ friction }) {
  return (
    <View style={styles.frictionCard}>
      <View style={styles.frictionSection}>
        <Text style={[styles.frictionLabel, { color: colors.red }]}>
          ECHTE WIDERSPRÜCHE
        </Text>
        <BulletList items={friction.echte_widersprueche} color={colors.red} />
      </View>

      <View style={[styles.frictionDivider, { backgroundColor: colors.border }]} />

      <View style={styles.frictionSection}>
        <Text style={[styles.frictionLabel, { color: colors.redDim }]}>
          ÜBERSETZUNGSFEHLER
        </Text>
        <BulletList items={friction.uebersetzungsfehler} color={colors.redDim} />
      </View>

      <View style={[styles.frictionDivider, { backgroundColor: colors.border }]} />

      <View style={styles.frictionSection}>
        <Text style={[styles.frictionLabel, { color: colors.textMuted }]}>
          KOLLEKTIV ÜBERSEHEN
        </Text>
        <Text style={styles.fieldValue}>{friction.uebersehenes}</Text>
      </View>
    </View>
  );
}

// ─── Einfach Gesagt ───────────────────────────────────────

function EinfachGesagtCard({ text }) {
  if (!text) return null;
  return (
    <View style={styles.einfachCard}>
      <Text style={[styles.frictionLabel, { color: colors.terracotta }]}>
        EINFACH GESAGT
      </Text>
      <Text style={styles.einfachText}>{text}</Text>
    </View>
  );
}

// ─── Herz & Kopf ──────────────────────────────────────────

function HerzKopfCard({ herzmensch, kopfmensch }) {
  if (!herzmensch && !kopfmensch) return null;
  return (
    <View style={styles.herzKopfRow}>
      {herzmensch ? (
        <View style={[styles.herzKopfCard, { borderColor: colors.heartDim, backgroundColor: colors.heartBg }]}>
          <Text style={[styles.herzKopfLabel, { color: colors.heart }]}>HERZMENSCH</Text>
          <Text style={styles.herzKopfText}>{herzmensch}</Text>
        </View>
      ) : null}
      {kopfmensch ? (
        <View style={[styles.herzKopfCard, { borderColor: colors.blueDim, backgroundColor: colors.blueBg }]}>
          <Text style={[styles.herzKopfLabel, { color: colors.blue }]}>KOPFMENSCH</Text>
          <Text style={styles.herzKopfText}>{kopfmensch}</Text>
        </View>
      ) : null}
    </View>
  );
}

// ─── Integrations-Karte ───────────────────────────────────
// Feldnamen: vorlaeufiges_fazit, uebersetzbare_bruecken, entscheidungshilfe,
//            echte_unvereinbarkeiten, praktische_optionen, offene_pruefpfade

function IntegrationCard({ integration }) {
  const sections = [
    {
      label:  "VORLÄUFIGES FAZIT",
      value:  integration.vorlaeufiges_fazit,
      isList: false,
      color:  colors.blue,
    },
    {
      label:  "ÜBERSETZBARE BRÜCKEN",
      value:  integration.uebersetzbare_bruecken,
      isList: true,
      color:  colors.blue,
    },
    {
      label:  "PRAKTISCHE OPTIONEN",
      value:  integration.praktische_optionen,
      isList: true,
      color:  colors.terracotta,
    },
    {
      label:  "ECHTE UNVEREINBARKEITEN",
      value:  integration.echte_unvereinbarkeiten,
      isList: true,
      color:  colors.red,
    },
    {
      label:  "ENTSCHEIDUNGSHILFE",
      value:  integration.entscheidungshilfe,
      isList: true,
      color:  colors.blueDim,
    },
    {
      label:  "OFFENE PRÜFPFADE",
      value:  integration.offene_pruefpfade,
      isList: true,
      color:  colors.textSecondary,
    },
  ];

  return (
    <View style={styles.integrationCard}>
      {sections.map((s, i) => {
        const isEmpty = s.isList ? (!s.value || s.value.length === 0) : !s.value;
        if (isEmpty) return null;
        return (
          <View key={i}>
            {i > 0 && <View style={[styles.frictionDivider, { backgroundColor: colors.borderSub }]} />}
            <View style={styles.frictionSection}>
              <Text style={[styles.frictionLabel, { color: s.color }]}>{s.label}</Text>
              {s.isList
                ? <BulletList items={s.value} color={s.color} />
                : <Text style={styles.fieldValue}>{s.value}</Text>
              }
            </View>
          </View>
        );
      })}
    </View>
  );
}

// ─── Haupt-Screen ─────────────────────────────────────────

export default function ResultScreen({ route, navigation }) {
  const { question, result } = route.params;
  const { perspectives, friction, integration } = result;

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <StatusBar barStyle="light-content" backgroundColor={colors.bg} />

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Text style={styles.backText}>← Neue Frage</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>FAMiLiEN TiSCH</Text>
        <View style={{ width: 80 }} />
      </View>

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {/* Frage */}
        <View style={styles.questionCard}>
          <Text style={styles.questionLabel}>DIE FRAGE</Text>
          <Text style={styles.questionText}>„{question}"</Text>
        </View>

        {/* Phase I — Perspektiven */}
        <SectionHeader
          phase="Phase I"
          title="Die Perspektiven"
          subtitle="Vier Methoden denken gleichzeitig — keine davon hat das letzte Wort."
          color={colors.terracotta}
        />
        <View style={styles.perspectivesGrid}>
          {perspectives.map((p, i) => (
            <PerspectiveCard key={i} perspective={p} index={i} />
          ))}
        </View>

        {/* Phase II — Reibung */}
        <SectionHeader
          phase="Phase II"
          title="Reibung & Widerspruch"
          subtitle="Wo widersprechen sich die Perspektiven wirklich?"
          color={colors.red}
        />
        <FrictionCard friction={friction} />

        {/* Phase III — Meta-Synthese */}
        <SectionHeader
          phase="Phase III"
          title="Die Meta-Synthese"
          subtitle="Kein Kuschelkonsens. Ehrliche Orientierung."
          color={colors.blue}
        />

        {/* Einfach Gesagt — zugängliche Zusammenfassung zuerst */}
        <EinfachGesagtCard text={integration.einfach_gesagt} />

        {/* Herzmensch & Kopfmensch */}
        <HerzKopfCard
          herzmensch={integration.herzmensch}
          kopfmensch={integration.kopfmensch}
        />

        {/* Detaillierte Synthese */}
        <IntegrationCard integration={integration} />

        {/* CTA */}
        <TouchableOpacity
          style={styles.newQuestionBtn}
          onPress={() => navigation.goBack()}
          activeOpacity={0.8}
        >
          <Text style={styles.newQuestionText}>Neue Frage auf den Tisch legen</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:   { flex: 1, backgroundColor: colors.bg },
  scroll: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl },

  // Header
  header: {
    flexDirection:    "row",
    alignItems:       "center",
    justifyContent:   "space-between",
    paddingHorizontal: spacing.lg,
    paddingVertical:  spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backBtn:     { width: 80 },
  backText:    { color: colors.terracotta, fontSize: typography.small, fontWeight: "600" },
  headerTitle: { color: colors.textPrimary, fontSize: typography.h3, fontWeight: "700" },

  // Frage
  questionCard: {
    backgroundColor:  colors.bgCard,
    borderRadius:     radius.md,
    borderWidth:      1,
    borderColor:      colors.border,
    borderLeftWidth:  3,
    borderLeftColor:  colors.terracotta,
    padding:          spacing.md,
    marginTop:        spacing.lg,
    marginBottom:     spacing.xl,
  },
  questionLabel: {
    fontSize:     typography.xs,
    color:        colors.terracotta,
    letterSpacing: 1.5,
    fontWeight:   "600",
    marginBottom: spacing.xs,
  },
  questionText: {
    fontSize:  typography.body,
    color:     colors.textPrimary,
    fontStyle: "italic",
    lineHeight: 24,
  },

  // Section Header
  sectionHeader: { marginBottom: spacing.md, marginTop: spacing.sm },
  phaseLabel:    { fontSize: typography.xs, letterSpacing: 2, fontWeight: "700", marginBottom: spacing.xs },
  sectionTitle:  { fontSize: typography.h2, color: colors.textPrimary, fontWeight: "700", marginBottom: spacing.xs },
  sectionSubtitle: { fontSize: typography.small, color: colors.textSecondary, lineHeight: 20 },

  // Perspektiven
  perspectivesGrid: { gap: spacing.md, marginBottom: spacing.xl },
  perspectiveCard: {
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.border,
    overflow:        "hidden",
    flexDirection:   "row",
  },
  perspectiveAccentBar: { width: 3 },
  perspectiveContent: { flex: 1, padding: spacing.md, gap: spacing.sm },
  perspectiveRole: { fontSize: typography.xs, letterSpacing: 1.5, fontWeight: "700", marginBottom: spacing.xs },

  // Feldblock
  fieldBlock: { gap: 4 },
  fieldLabel: { fontSize: typography.xs, letterSpacing: 1, fontWeight: "600" },
  fieldValue: { fontSize: typography.small, color: colors.textSecondary, lineHeight: 22 },

  // Bullet
  bulletList: { gap: spacing.sm },
  bulletRow:  { flexDirection: "row", gap: spacing.sm, alignItems: "flex-start" },
  bullet:     { width: 5, height: 5, borderRadius: 2.5, marginTop: 8, flexShrink: 0 },
  bulletText: { flex: 1, fontSize: typography.small, color: colors.textSecondary, lineHeight: 22 },

  // Reibung
  frictionCard: {
    backgroundColor: colors.redBg,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.redDim,
    overflow:        "hidden",
    marginBottom:    spacing.xl,
  },
  frictionSection: { padding: spacing.md, gap: spacing.sm },
  frictionDivider: { height: 1, marginHorizontal: spacing.md },
  frictionLabel:   { fontSize: typography.xs, letterSpacing: 1.5, fontWeight: "700" },

  // Einfach Gesagt
  einfachCard: {
    backgroundColor: colors.terracottaBg,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.terracottaDim,
    padding:         spacing.md,
    marginBottom:    spacing.md,
    gap:             spacing.sm,
  },
  einfachText: { fontSize: typography.body, color: colors.textPrimary, lineHeight: 26 },

  // Herz & Kopf
  herzKopfRow: { flexDirection: "row", gap: spacing.sm, marginBottom: spacing.md },
  herzKopfCard: {
    flex:         1,
    borderRadius: radius.md,
    borderWidth:  1,
    padding:      spacing.md,
    gap:          spacing.sm,
  },
  herzKopfLabel: { fontSize: typography.xs, letterSpacing: 1.5, fontWeight: "700" },
  herzKopfText:  { fontSize: typography.small, color: colors.textSecondary, lineHeight: 20 },

  // Integration
  integrationCard: {
    backgroundColor: colors.blueBg,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.blueDim,
    overflow:        "hidden",
    marginBottom:    spacing.xl,
  },

  // CTA
  newQuestionBtn: {
    backgroundColor: colors.bgCard,
    borderRadius:    radius.md,
    borderWidth:     1,
    borderColor:     colors.terracotta,
    paddingVertical: spacing.md,
    alignItems:      "center",
    marginTop:       spacing.sm,
  },
  newQuestionText: { color: colors.terracotta, fontWeight: "600", fontSize: typography.body },
});
