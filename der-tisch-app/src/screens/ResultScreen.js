import React, { useRef, useEffect } from "react";
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  Animated, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { colors, spacing, radius, typography, agentColors } from "../theme";

// ─── Small Components ─────────────────────────────────────

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
  if (!value) return null;
  return (
    <View style={styles.fieldBlock}>
      <Text style={[styles.fieldLabel, { color: color || colors.textMuted }]}>{label}</Text>
      <Text style={styles.fieldValue}>{value}</Text>
    </View>
  );
}

function BulletList({ items, color }) {
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

// ─── Perspective Card ─────────────────────────────────────

function PerspectiveCard({ perspective, index }) {
  const ac = agentColors[perspective.rolle] || { accent: colors.amber, bg: colors.bgCard };
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 400,
      delay: index * 120,
      useNativeDriver: true,
    }).start();
  }, []);

  return (
    <Animated.View style={[styles.perspectiveCard, { backgroundColor: ac.bg, opacity: fadeAnim }]}>
      <View style={[styles.perspectiveAccentBar, { backgroundColor: ac.accent }]} />
      <View style={styles.perspectiveContent}>
        <Text style={[styles.perspectiveRole, { color: ac.accent }]}>
          {perspective.rolle.toUpperCase()}
        </Text>
        <FieldBlock label="Kernanalyse" value={perspective.kernanalyse} color={ac.accent} />
        <FieldBlock label="Evidenz" value={perspective.evidenz} color={ac.accent} />
        <FieldBlock label="Blinder Fleck" value={perspective.blinder_fleck} color={ac.accent} />
      </View>
    </Animated.View>
  );
}

// ─── Friction Card ────────────────────────────────────────

function FrictionCard({ friction }) {
  return (
    <View style={styles.frictionCard}>
      <View style={styles.frictionSection}>
        <Text style={[styles.frictionLabel, { color: colors.red }]}>
          HARTE WIDERSPRÜCHE
        </Text>
        <BulletList items={friction.harte_widersprueche} color={colors.red} />
      </View>

      <View style={[styles.frictionDivider, { backgroundColor: colors.border }]} />

      <View style={styles.frictionSection}>
        <Text style={[styles.frictionLabel, { color: colors.redDim }]}>
          SCHEINKONSENS
        </Text>
        <BulletList items={friction.scheinkonsens} color={colors.redDim} />
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

// ─── Integration Card ─────────────────────────────────────

function IntegrationCard({ integration }) {
  const sections = [
    {
      label: "VORLÄUFIGER KONSENS",
      value: integration.vorlaeufiger_konsens,
      isList: false,
      color: colors.blue,
    },
    {
      label: "FRUCHTBARE DIFFERENZEN",
      value: integration.fruchtbare_differenzen,
      isList: true,
      color: colors.blue,
    },
    {
      label: "ÜBERSETZBARKEIT",
      value: integration.uebersetzbarkeit,
      isList: true,
      color: colors.blueDim,
    },
    {
      label: "ECHTE UNVEREINBARKEITEN",
      value: integration.echte_unvereinbarkeiten,
      isList: true,
      color: colors.red,
    },
    {
      label: "PRAKTISCHE OPTIONEN",
      value: integration.praktische_optionen,
      isList: true,
      color: colors.amber,
    },
    {
      label: "OFFENE PRÜFPFADE",
      value: integration.offene_pruefpfade,
      isList: true,
      color: colors.textSecondary,
    },
  ];

  return (
    <View style={styles.integrationCard}>
      {sections.map((s, i) => (
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
      ))}
    </View>
  );
}

// ─── Main Screen ──────────────────────────────────────────

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
        {/* Question */}
        <View style={styles.questionCard}>
          <Text style={styles.questionLabel}>DIE FRAGE</Text>
          <Text style={styles.questionText}>„{question}"</Text>
        </View>

        {/* Phase I — Perspektiven */}
        <SectionHeader
          phase="Phase I"
          title="Die Perspektiven"
          subtitle="Vier Methoden denken gleichzeitig — keine davon hat recht."
          color={colors.amber}
        />
        <View style={styles.perspectivesGrid}>
          {perspectives.map((p, i) => (
            <PerspectiveCard key={i} perspective={p} index={i} />
          ))}
        </View>

        {/* Phase II — Reibung */}
        <SectionHeader
          phase="Phase II"
          title="Reibung & Scheinkonsens"
          subtitle="Wo widersprechen sich die Methoden wirklich?"
          color={colors.red}
        />
        <FrictionCard friction={friction} />

        {/* Phase III — Integration */}
        <SectionHeader
          phase="Phase III"
          title="Die Meta-Synthese"
          subtitle="Kein Kuschelkonsens. Ehrliche Integration."
          color={colors.blue}
        />
        <IntegrationCard integration={integration} />

        {/* Bottom CTA */}
        <TouchableOpacity
          style={styles.newQuestionBtn}
          onPress={() => navigation.goBack()}
          activeOpacity={0.8}
        >
          <Text style={styles.newQuestionText}>Neue Frage an den TiSCH</Text>
        </TouchableOpacity>
      </ScrollView>
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

  // Header
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backBtn: {
    width: 80,
  },
  backText: {
    color: colors.amber,
    fontSize: typography.small,
    fontWeight: "600",
  },
  headerTitle: {
    color: colors.textPrimary,
    fontSize: typography.h3,
    fontWeight: "700",
  },

  // Question
  questionCard: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    borderLeftWidth: 3,
    borderLeftColor: colors.amber,
    padding: spacing.md,
    marginTop: spacing.lg,
    marginBottom: spacing.xl,
  },
  questionLabel: {
    fontSize: typography.xs,
    color: colors.amber,
    letterSpacing: 1.5,
    fontWeight: "600",
    marginBottom: spacing.xs,
  },
  questionText: {
    fontSize: typography.body,
    color: colors.textPrimary,
    fontStyle: "italic",
    lineHeight: 24,
  },

  // Section headers
  sectionHeader: {
    marginBottom: spacing.md,
    marginTop: spacing.sm,
  },
  phaseLabel: {
    fontSize: typography.xs,
    letterSpacing: 2,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  sectionTitle: {
    fontSize: typography.h2,
    color: colors.textPrimary,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  sectionSubtitle: {
    fontSize: typography.small,
    color: colors.textSecondary,
    lineHeight: 20,
  },

  // Perspectives grid
  perspectivesGrid: {
    gap: spacing.md,
    marginBottom: spacing.xl,
  },
  perspectiveCard: {
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: "hidden",
    flexDirection: "row",
  },
  perspectiveAccentBar: {
    width: 3,
  },
  perspectiveContent: {
    flex: 1,
    padding: spacing.md,
    gap: spacing.sm,
  },
  perspectiveRole: {
    fontSize: typography.xs,
    letterSpacing: 1.5,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },

  // Field block
  fieldBlock: {
    gap: 4,
  },
  fieldLabel: {
    fontSize: typography.xs,
    letterSpacing: 1,
    fontWeight: "600",
  },
  fieldValue: {
    fontSize: typography.small,
    color: colors.textSecondary,
    lineHeight: 22,
  },

  // Bullet list
  bulletList: {
    gap: spacing.sm,
  },
  bulletRow: {
    flexDirection: "row",
    gap: spacing.sm,
    alignItems: "flex-start",
  },
  bullet: {
    width: 5,
    height: 5,
    borderRadius: 2.5,
    marginTop: 8,
    flexShrink: 0,
  },
  bulletText: {
    flex: 1,
    fontSize: typography.small,
    color: colors.textSecondary,
    lineHeight: 22,
  },

  // Friction card
  frictionCard: {
    backgroundColor: colors.redBg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.redDim,
    overflow: "hidden",
    marginBottom: spacing.xl,
  },
  frictionSection: {
    padding: spacing.md,
    gap: spacing.sm,
  },
  frictionDivider: {
    height: 1,
    marginHorizontal: spacing.md,
  },
  frictionLabel: {
    fontSize: typography.xs,
    letterSpacing: 1.5,
    fontWeight: "700",
  },

  // Integration card
  integrationCard: {
    backgroundColor: colors.blueBg,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.blueDim,
    overflow: "hidden",
    marginBottom: spacing.xl,
  },

  // Bottom CTA
  newQuestionBtn: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.amber,
    paddingVertical: spacing.md,
    alignItems: "center",
    marginTop: spacing.sm,
  },
  newQuestionText: {
    color: colors.amber,
    fontWeight: "600",
    fontSize: typography.body,
  },
});
