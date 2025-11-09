package trustformers

/*
#include <stdlib.h>

// Pipeline-related functions
extern void* trustformers_create_text_generation_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern void* trustformers_create_text_classification_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern void* trustformers_create_question_answering_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern void* trustformers_create_conversational_pipeline(void* model, void* tokenizer, TrustformersError* error);
extern TrustformersError trustformers_pipeline_free(void* pipeline);

// Text generation
extern TrustformersError trustformers_pipeline_generate_text(void* pipeline, const char* prompt, char** generated_text);
extern TrustformersError trustformers_pipeline_generate_text_with_options(void* pipeline, const char* prompt, const char* options_json, char** generated_text);
extern TrustformersError trustformers_pipeline_generate_text_streaming(void* pipeline, const char* prompt, void* callback, void* user_data);

// Text classification
extern TrustformersError trustformers_pipeline_classify_text(void* pipeline, const char* text, char** classification_result);
extern TrustformersError trustformers_pipeline_classify_text_batch(void* pipeline, const char** texts, int num_texts, char** classification_results);

// Question answering
extern TrustformersError trustformers_pipeline_answer_question(void* pipeline, const char* context, const char* question, char** answer);

// Conversational
extern TrustformersError trustformers_pipeline_add_turn(void* pipeline, const char* user_input, char** bot_response);
extern TrustformersError trustformers_pipeline_get_conversation_history(void* pipeline, char** history_json);
extern TrustformersError trustformers_pipeline_clear_conversation(void* pipeline);

// Common pipeline functions
extern TrustformersError trustformers_pipeline_get_info(void* pipeline, char** info_json);
extern TrustformersError trustformers_pipeline_set_options(void* pipeline, const char* options_json);
extern TrustformersError trustformers_pipeline_get_performance_stats(void* pipeline, char** stats_json);
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// PipelineType represents the type of pipeline
type PipelineType string

const (
	TextGeneration     PipelineType = "text-generation"
	TextClassification PipelineType = "text-classification"
	QuestionAnswering  PipelineType = "question-answering"
	Conversational     PipelineType = "conversational"
)

// Pipeline represents a machine learning pipeline
type Pipeline struct {
	handle      unsafe.Pointer
	pipelineType PipelineType
	tf          *TrustformeRS
}

// GenerationOptions contains options for text generation
type GenerationOptions struct {
	MaxLength        int     `json:"max_length,omitempty"`
	MinLength        int     `json:"min_length,omitempty"`
	Temperature      float64 `json:"temperature,omitempty"`
	TopK             int     `json:"top_k,omitempty"`
	TopP             float64 `json:"top_p,omitempty"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitempty"`
	DoSample         bool    `json:"do_sample,omitempty"`
	EarlyStopping    bool    `json:"early_stopping,omitempty"`
	NumBeams         int     `json:"num_beams,omitempty"`
	NumReturnSequences int   `json:"num_return_sequences,omitempty"`
}

// ClassificationResult represents text classification results
type ClassificationResult struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// AnswerResult represents question answering results
type AnswerResult struct {
	Answer    string  `json:"answer"`
	Score     float64 `json:"score"`
	Start     int     `json:"start"`
	End       int     `json:"end"`
}

// ConversationTurn represents a conversation turn
type ConversationTurn struct {
	UserInput   string `json:"user_input"`
	BotResponse string `json:"bot_response"`
	Timestamp   int64  `json:"timestamp"`
}

// PipelineInfo contains pipeline information
type PipelineInfo struct {
	Type        string                 `json:"type"`
	ModelName   string                 `json:"model_name"`
	TokenizerName string               `json:"tokenizer_name"`
	Capabilities []string              `json:"capabilities"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// StreamingCallback is the type for streaming text generation callbacks
type StreamingCallback func(text string, done bool, userData interface{})

// CreateTextGenerationPipeline creates a text generation pipeline
func (tf *TrustformeRS) CreateTextGenerationPipeline(model *Model, tokenizer *Tokenizer) (*Pipeline, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	if model == nil || model.handle == nil {
		return nil, errors.New("invalid model")
	}

	if tokenizer == nil || tokenizer.handle == nil {
		return nil, errors.New("invalid tokenizer")
	}

	var cError C.TrustformersError
	handle := C.trustformers_create_text_generation_pipeline(model.handle, tokenizer.handle, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	pipeline := &Pipeline{
		handle:      handle,
		pipelineType: TextGeneration,
		tf:          tf,
	}

	runtime.SetFinalizer(pipeline, (*Pipeline).finalize)
	return pipeline, nil
}

// CreateTextClassificationPipeline creates a text classification pipeline
func (tf *TrustformeRS) CreateTextClassificationPipeline(model *Model, tokenizer *Tokenizer) (*Pipeline, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	if model == nil || model.handle == nil {
		return nil, errors.New("invalid model")
	}

	if tokenizer == nil || tokenizer.handle == nil {
		return nil, errors.New("invalid tokenizer")
	}

	var cError C.TrustformersError
	handle := C.trustformers_create_text_classification_pipeline(model.handle, tokenizer.handle, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	pipeline := &Pipeline{
		handle:      handle,
		pipelineType: TextClassification,
		tf:          tf,
	}

	runtime.SetFinalizer(pipeline, (*Pipeline).finalize)
	return pipeline, nil
}

// CreateQuestionAnsweringPipeline creates a question answering pipeline
func (tf *TrustformeRS) CreateQuestionAnsweringPipeline(model *Model, tokenizer *Tokenizer) (*Pipeline, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	if model == nil || model.handle == nil {
		return nil, errors.New("invalid model")
	}

	if tokenizer == nil || tokenizer.handle == nil {
		return nil, errors.New("invalid tokenizer")
	}

	var cError C.TrustformersError
	handle := C.trustformers_create_question_answering_pipeline(model.handle, tokenizer.handle, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	pipeline := &Pipeline{
		handle:      handle,
		pipelineType: QuestionAnswering,
		tf:          tf,
	}

	runtime.SetFinalizer(pipeline, (*Pipeline).finalize)
	return pipeline, nil
}

// CreateConversationalPipeline creates a conversational pipeline
func (tf *TrustformeRS) CreateConversationalPipeline(model *Model, tokenizer *Tokenizer) (*Pipeline, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	if model == nil || model.handle == nil {
		return nil, errors.New("invalid model")
	}

	if tokenizer == nil || tokenizer.handle == nil {
		return nil, errors.New("invalid tokenizer")
	}

	var cError C.TrustformersError
	handle := C.trustformers_create_conversational_pipeline(model.handle, tokenizer.handle, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	pipeline := &Pipeline{
		handle:      handle,
		pipelineType: Conversational,
		tf:          tf,
	}

	runtime.SetFinalizer(pipeline, (*Pipeline).finalize)
	return pipeline, nil
}

// Free releases the pipeline resources
func (p *Pipeline) Free() error {
	if p.handle == nil {
		return nil
	}

	err := C.trustformers_pipeline_free(p.handle)
	if err := checkError(err); err != nil {
		return err
	}

	p.handle = nil
	runtime.SetFinalizer(p, nil)
	return nil
}

// finalize is called by the finalizer
func (p *Pipeline) finalize() {
	if p.handle != nil {
		p.Free()
	}
}

// GenerateText generates text from a prompt (for text generation pipelines)
func (p *Pipeline) GenerateText(prompt string) (string, error) {
	if p.handle == nil {
		return "", errors.New("pipeline not loaded")
	}

	if p.pipelineType != TextGeneration {
		return "", errors.New("not a text generation pipeline")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cGeneratedText *C.char
	err := C.trustformers_pipeline_generate_text(p.handle, cPrompt, &cGeneratedText)
	if err := checkError(err); err != nil {
		return "", err
	}
	defer freeCString(cGeneratedText)

	if cGeneratedText == nil {
		return "", nil
	}

	return C.GoString(cGeneratedText), nil
}

// GenerateTextWithOptions generates text with custom options
func (p *Pipeline) GenerateTextWithOptions(prompt string, options GenerationOptions) (string, error) {
	if p.handle == nil {
		return "", errors.New("pipeline not loaded")
	}

	if p.pipelineType != TextGeneration {
		return "", errors.New("not a text generation pipeline")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	optionsJSON, err := json.Marshal(options)
	if err != nil {
		return "", err
	}

	cOptionsJSON := C.CString(string(optionsJSON))
	defer C.free(unsafe.Pointer(cOptionsJSON))

	var cGeneratedText *C.char
	err2 := C.trustformers_pipeline_generate_text_with_options(p.handle, cPrompt, cOptionsJSON, &cGeneratedText)
	if err := checkError(err2); err != nil {
		return "", err
	}
	defer freeCString(cGeneratedText)

	if cGeneratedText == nil {
		return "", nil
	}

	return C.GoString(cGeneratedText), nil
}

// ClassifyText classifies text (for text classification pipelines)
func (p *Pipeline) ClassifyText(text string) ([]ClassificationResult, error) {
	if p.handle == nil {
		return nil, errors.New("pipeline not loaded")
	}

	if p.pipelineType != TextClassification {
		return nil, errors.New("not a text classification pipeline")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var cClassificationResult *C.char
	err := C.trustformers_pipeline_classify_text(p.handle, cText, &cClassificationResult)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cClassificationResult)

	if cClassificationResult == nil {
		return []ClassificationResult{}, nil
	}

	resultJSON := C.GoString(cClassificationResult)
	var results []ClassificationResult
	if err := json.Unmarshal([]byte(resultJSON), &results); err != nil {
		return nil, err
	}

	return results, nil
}

// ClassifyTextBatch classifies multiple texts
func (p *Pipeline) ClassifyTextBatch(texts []string) ([][]ClassificationResult, error) {
	if p.handle == nil {
		return nil, errors.New("pipeline not loaded")
	}

	if p.pipelineType != TextClassification {
		return nil, errors.New("not a text classification pipeline")
	}

	if len(texts) == 0 {
		return [][]ClassificationResult{}, nil
	}

	// Convert Go string slice to C array
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}
	defer func() {
		for _, cText := range cTexts {
			C.free(unsafe.Pointer(cText))
		}
	}()

	var cClassificationResults *C.char
	err := C.trustformers_pipeline_classify_text_batch(p.handle, &cTexts[0], C.int(len(texts)), &cClassificationResults)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cClassificationResults)

	if cClassificationResults == nil {
		return make([][]ClassificationResult, len(texts)), nil
	}

	resultsJSON := C.GoString(cClassificationResults)
	var results [][]ClassificationResult
	if err := json.Unmarshal([]byte(resultsJSON), &results); err != nil {
		return nil, err
	}

	return results, nil
}

// AnswerQuestion answers a question given context (for question answering pipelines)
func (p *Pipeline) AnswerQuestion(context, question string) (AnswerResult, error) {
	if p.handle == nil {
		return AnswerResult{}, errors.New("pipeline not loaded")
	}

	if p.pipelineType != QuestionAnswering {
		return AnswerResult{}, errors.New("not a question answering pipeline")
	}

	cContext := C.CString(context)
	defer C.free(unsafe.Pointer(cContext))

	cQuestion := C.CString(question)
	defer C.free(unsafe.Pointer(cQuestion))

	var cAnswer *C.char
	err := C.trustformers_pipeline_answer_question(p.handle, cContext, cQuestion, &cAnswer)
	if err := checkError(err); err != nil {
		return AnswerResult{}, err
	}
	defer freeCString(cAnswer)

	if cAnswer == nil {
		return AnswerResult{}, nil
	}

	answerJSON := C.GoString(cAnswer)
	var result AnswerResult
	if err := json.Unmarshal([]byte(answerJSON), &result); err != nil {
		return AnswerResult{}, err
	}

	return result, nil
}

// AddTurn adds a conversation turn (for conversational pipelines)
func (p *Pipeline) AddTurn(userInput string) (string, error) {
	if p.handle == nil {
		return "", errors.New("pipeline not loaded")
	}

	if p.pipelineType != Conversational {
		return "", errors.New("not a conversational pipeline")
	}

	cUserInput := C.CString(userInput)
	defer C.free(unsafe.Pointer(cUserInput))

	var cBotResponse *C.char
	err := C.trustformers_pipeline_add_turn(p.handle, cUserInput, &cBotResponse)
	if err := checkError(err); err != nil {
		return "", err
	}
	defer freeCString(cBotResponse)

	if cBotResponse == nil {
		return "", nil
	}

	return C.GoString(cBotResponse), nil
}

// GetConversationHistory returns the conversation history
func (p *Pipeline) GetConversationHistory() ([]ConversationTurn, error) {
	if p.handle == nil {
		return nil, errors.New("pipeline not loaded")
	}

	if p.pipelineType != Conversational {
		return nil, errors.New("not a conversational pipeline")
	}

	var cHistoryJSON *C.char
	err := C.trustformers_pipeline_get_conversation_history(p.handle, &cHistoryJSON)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cHistoryJSON)

	if cHistoryJSON == nil {
		return []ConversationTurn{}, nil
	}

	historyJSON := C.GoString(cHistoryJSON)
	var history []ConversationTurn
	if err := json.Unmarshal([]byte(historyJSON), &history); err != nil {
		return nil, err
	}

	return history, nil
}

// ClearConversation clears the conversation history
func (p *Pipeline) ClearConversation() error {
	if p.handle == nil {
		return errors.New("pipeline not loaded")
	}

	if p.pipelineType != Conversational {
		return errors.New("not a conversational pipeline")
	}

	err := C.trustformers_pipeline_clear_conversation(p.handle)
	return checkError(err)
}

// GetInfo returns pipeline information
func (p *Pipeline) GetInfo() (PipelineInfo, error) {
	if p.handle == nil {
		return PipelineInfo{}, errors.New("pipeline not loaded")
	}

	var cInfoJSON *C.char
	err := C.trustformers_pipeline_get_info(p.handle, &cInfoJSON)
	if err := checkError(err); err != nil {
		return PipelineInfo{}, err
	}
	defer freeCString(cInfoJSON)

	if cInfoJSON == nil {
		return PipelineInfo{}, errors.New("failed to get pipeline info")
	}

	infoJSON := C.GoString(cInfoJSON)
	var info PipelineInfo
	if err := json.Unmarshal([]byte(infoJSON), &info); err != nil {
		return PipelineInfo{}, err
	}

	return info, nil
}

// SetOptions sets pipeline options
func (p *Pipeline) SetOptions(options map[string]interface{}) error {
	if p.handle == nil {
		return errors.New("pipeline not loaded")
	}

	optionsJSON, err := json.Marshal(options)
	if err != nil {
		return err
	}

	cOptionsJSON := C.CString(string(optionsJSON))
	defer C.free(unsafe.Pointer(cOptionsJSON))

	err2 := C.trustformers_pipeline_set_options(p.handle, cOptionsJSON)
	return checkError(err2)
}

// GetPerformanceStats returns pipeline performance statistics
func (p *Pipeline) GetPerformanceStats() (map[string]interface{}, error) {
	if p.handle == nil {
		return nil, errors.New("pipeline not loaded")
	}

	var cStatsJSON *C.char
	err := C.trustformers_pipeline_get_performance_stats(p.handle, &cStatsJSON)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer freeCString(cStatsJSON)

	if cStatsJSON == nil {
		return make(map[string]interface{}), nil
	}

	statsJSON := C.GoString(cStatsJSON)
	var stats map[string]interface{}
	if err := json.Unmarshal([]byte(statsJSON), &stats); err != nil {
		return nil, err
	}

	return stats, nil
}

// GetType returns the pipeline type
func (p *Pipeline) GetType() PipelineType {
	return p.pipelineType
}

// IsLoaded checks if the pipeline is loaded
func (p *Pipeline) IsLoaded() bool {
	return p.handle != nil
}