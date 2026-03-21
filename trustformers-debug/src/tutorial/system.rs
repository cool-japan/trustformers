//! Interactive tutorial and learning system for TrustformeRS debugging
//!
//! This module provides educational content, guided tutorials, and progressive
//! lessons to help users learn debugging concepts and best practices.

use anyhow::Result;

/// Interactive tutorial system
pub struct TutorialMode {
    current_lesson: usize,
    lessons: Vec<TutorialLesson>,
    completed_lessons: Vec<bool>,
}

/// Tutorial lesson
#[derive(Debug, Clone)]
pub struct TutorialLesson {
    pub title: String,
    pub description: String,
    pub objectives: Vec<String>,
    pub example_code: String,
    pub expected_output: String,
    pub tips: Vec<String>,
    pub common_mistakes: Vec<String>,
}

impl Default for TutorialMode {
    fn default() -> Self {
        Self::new()
    }
}

impl TutorialMode {
    /// Create new tutorial mode
    pub fn new() -> Self {
        let lessons = create_tutorial_lessons();
        let completed_lessons = vec![false; lessons.len()];

        Self {
            current_lesson: 0,
            lessons,
            completed_lessons,
        }
    }

    /// Get current lesson
    pub fn current_lesson(&self) -> Option<&TutorialLesson> {
        self.lessons.get(self.current_lesson)
    }

    /// Get lesson by index
    pub fn get_lesson(&self, index: usize) -> Option<&TutorialLesson> {
        self.lessons.get(index)
    }

    /// Get total number of lessons
    pub fn total_lessons(&self) -> usize {
        self.lessons.len()
    }

    /// Get completion progress
    pub fn progress(&self) -> f64 {
        let completed = self.completed_lessons.iter().filter(|&&x| x).count();
        (completed as f64 / self.total_lessons() as f64) * 100.0
    }

    /// Mark current lesson as completed and move to next
    pub fn complete_current_lesson(&mut self) -> Result<()> {
        if self.current_lesson < self.total_lessons() {
            self.completed_lessons[self.current_lesson] = true;
            self.current_lesson += 1;
            Ok(())
        } else {
            Err(anyhow::anyhow!("No more lessons to complete"))
        }
    }

    /// Navigate to specific lesson
    pub fn goto_lesson(&mut self, index: usize) -> Result<()> {
        if index < self.total_lessons() {
            self.current_lesson = index;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid lesson index"))
        }
    }

    /// Check if tutorial is complete
    pub fn is_complete(&self) -> bool {
        self.completed_lessons.iter().all(|&x| x)
    }

    /// Get lesson titles for navigation
    pub fn lesson_titles(&self) -> Vec<String> {
        self.lessons.iter().map(|l| l.title.clone()).collect()
    }
}

/// Create tutorial lessons
fn create_tutorial_lessons() -> Vec<TutorialLesson> {
    vec![
        TutorialLesson {
            title: "Getting Started with TrustformeRS Debug".to_string(),
            description: "Learn the basics of debugging with TrustformeRS".to_string(),
            objectives: vec![
                "Create a debug session".to_string(),
                "Perform basic model inspection".to_string(),
                "Generate a simple report".to_string(),
            ],
            example_code: r#"
use trustformers_debug::*;

// Create a debug session with default configuration
let mut session = debug_session();

// Start debugging
session.start().await?;

// Your model operations here...

// Generate report
let report = session.stop().await?;
println!("Debug report: {:?}", report.summary());
"#
            .to_string(),
            expected_output: "Debug session should start successfully and generate a report"
                .to_string(),
            tips: vec![
                "Always call start() before debugging operations".to_string(),
                "Use stop() to generate final report".to_string(),
                "Check the summary for quick insights".to_string(),
            ],
            common_mistakes: vec![
                "Forgetting to call start() before debugging".to_string(),
                "Not handling async operations properly".to_string(),
            ],
        },
        TutorialLesson {
            title: "One-Line Debugging".to_string(),
            description: "Use simplified debugging functions for quick analysis".to_string(),
            objectives: vec![
                "Use the debug() function".to_string(),
                "Understand different debug levels".to_string(),
                "Interpret simplified results".to_string(),
            ],
            example_code: r#"
use trustformers_debug::*;

// Quick debugging with automatic level detection
let result = debug(&model).await?;
println!("Summary: {}", result.summary());

// Check for critical issues
if result.has_critical_issues() {
    println!("Critical issues found!");
    for rec in result.recommendations() {
        println!("- {}", rec);
    }
}

// Use specific debug levels
let light_result = quick_debug(&model, QuickDebugLevel::Light).await?;
let deep_result = quick_debug(&model, QuickDebugLevel::Deep).await?;
"#
            .to_string(),
            expected_output: "Quick analysis results with recommendations".to_string(),
            tips: vec![
                "Use Light level for quick checks".to_string(),
                "Use Deep level for thorough analysis".to_string(),
                "Check recommendations for actionable advice".to_string(),
            ],
            common_mistakes: vec![
                "Using Deep level when Light would suffice".to_string(),
                "Ignoring critical issue warnings".to_string(),
            ],
        },
        TutorialLesson {
            title: "Guided Debugging".to_string(),
            description: "Step-by-step debugging with the guided debugger".to_string(),
            objectives: vec![
                "Use the guided debugger".to_string(),
                "Execute debugging steps".to_string(),
                "Understand step results".to_string(),
            ],
            example_code: r#"
use trustformers_debug::*;

// Create guided debugger
let mut guided = GuidedDebugger::new();

// Execute steps one by one
while !guided.is_complete() {
    if let Some(step) = guided.current_step() {
        println!("Step {}/{}: {}",
            guided.current_step + 1,
            guided.total_steps(),
            step.name
        );
        println!("Description: {}", step.description);

        // Execute the step
        let result = guided.execute_current_step().await?;
        println!("Result: {:?}", result);
    }
}

println!("Debugging complete! Progress: {:.1}%", guided.progress());
"#
            .to_string(),
            expected_output: "Step-by-step execution with progress updates".to_string(),
            tips: vec![
                "Read step descriptions carefully".to_string(),
                "You can skip steps if not relevant".to_string(),
                "Monitor progress percentage".to_string(),
            ],
            common_mistakes: vec![
                "Skipping important steps".to_string(),
                "Not reading step descriptions".to_string(),
            ],
        },
    ]
}
