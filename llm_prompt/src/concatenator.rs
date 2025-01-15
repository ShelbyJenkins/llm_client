use serde::{Deserialize, Serialize};

/// Controls how text segments are joined together in prompt messages.
///
/// `TextConcatenator` provides standard and custom options for combining multiple
/// text segments within a prompt message. This enum offers common concatenation
/// patterns like newlines and spaces, as well as the ability to specify custom
/// separators.
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
pub enum TextConcatenator {
    /// Joins text segments with double newlines ("\n\n")
    DoubleNewline,

    /// Joins text segments with a single newline ("\n").
    /// This is the default concatenation method.
    #[default]
    SingleNewline,

    /// Joins text segments with a single space
    Space,

    /// Joins text segments with a comma followed by a space (", ")
    Comma,

    /// Joins text segments with a custom separator string
    Custom(String),
}

impl TextConcatenator {
    /// Returns the separator string for this concatenator.
    ///
    /// # Returns
    ///
    /// A string slice containing the separatorfs
    pub fn as_str(&self) -> &str {
        match self {
            TextConcatenator::DoubleNewline => "\n\n",
            TextConcatenator::SingleNewline => "\n",
            TextConcatenator::Space => " ",
            TextConcatenator::Comma => ", ",
            TextConcatenator::Custom(custom) => custom,
        }
    }
}

/// Provides methods for managing text concatenation behavior.
///
/// This trait defines a standard interface for types that need to control how
/// their text segments are joined together. It offers methods to switch between
/// different concatenation styles and ensures proper state management when
/// concatenation rules change.
pub trait TextConcatenatorTrait {
    /// Provides mutable access to the concatenator.
    fn concatenator_mut(&mut self) -> &mut TextConcatenator;

    /// Clears any built content when concatenation rules change.
    fn clear_built(&self);

    /// Sets double newline concatenation ("\n\n").
    ///
    /// Changes the concatenator to use double newlines and clears built content
    /// if the concatenation style has changed.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    fn concate_deol(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::DoubleNewline {
            *self.concatenator_mut() = TextConcatenator::DoubleNewline;
            self.clear_built();
        }
        self
    }

    /// Sets single newline concatenation ("\n").
    ///
    /// Changes the concatenator to use single newlines and clears built content
    /// if the concatenation style has changed.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    fn concate_seol(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::SingleNewline {
            *self.concatenator_mut() = TextConcatenator::SingleNewline;
            self.clear_built();
        }
        self
    }

    /// Sets space concatenation (" ").
    ///
    /// Changes the concatenator to use spaces and clears built content
    /// if the concatenation style has changed.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    fn concate_space(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Space {
            *self.concatenator_mut() = TextConcatenator::Space;
            self.clear_built();
        }
        self
    }

    /// Sets comma concatenation (", ").
    ///
    /// Changes the concatenator to use commas followed by spaces and clears
    /// built content if the concatenation style has changed.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    fn concate_comma(&mut self) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Comma {
            *self.concatenator_mut() = TextConcatenator::Comma;
            self.clear_built();
        }
        self
    }

    /// Sets custom concatenation with the provided separator.
    ///
    /// Changes the concatenator to use a custom separator string and clears
    /// built content if the concatenation style has changed.
    ///
    /// # Arguments
    ///
    /// * `custom` - The custom separator string to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    fn concate_custom<T: AsRef<str>>(&mut self, custom: T) -> &mut Self {
        if self.concatenator_mut() != &TextConcatenator::Custom(custom.as_ref().to_owned()) {
            *self.concatenator_mut() = TextConcatenator::Custom(custom.as_ref().to_owned());
            self.clear_built();
        }
        self
    }
}
