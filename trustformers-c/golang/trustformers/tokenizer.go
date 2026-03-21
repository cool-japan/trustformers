package trustformers

/*
#include <stdlib.h>

// Tokenizer-related functions
extern void* trustformers_load_tokenizer_from_hub(const char* model_name, TrustformersError* error);
extern void* trustformers_load_tokenizer_from_path(const char* tokenizer_path, TrustformersError* error);
extern TrustformersError trustformers_tokenizer_free(void* tokenizer);
extern TrustformersError trustformers_tokenizer_encode(void* tokenizer, const char* text, int** tokens, int* num_tokens);
extern TrustformersError trustformers_tokenizer_decode(void* tokenizer, const int* tokens, int num_tokens, char** text);
extern TrustformersError trustformers_tokenizer_encode_batch(void* tokenizer, const char** texts, int num_texts, int*** tokens, int** num_tokens);
extern TrustformersError trustformers_tokenizer_decode_batch(void* tokenizer, const int** tokens, const int* num_tokens, int batch_size, char*** texts);
extern TrustformersError trustformers_tokenizer_get_vocab_size(void* tokenizer, int* vocab_size);
extern TrustformersError trustformers_tokenizer_get_special_tokens(void* tokenizer, char** special_tokens_json);
extern TrustformersError trustformers_tokenizer_add_special_token(void* tokenizer, const char* token, int token_id);
extern TrustformersError trustformers_tokenizer_get_info(void* tokenizer, char** info_json);
extern void trustformers_free_tokens(int* tokens);
extern void trustformers_free_token_batches(int** tokens, int batch_size);
extern void trustformers_free_text_batch(char** texts, int batch_size);
*/
import "C"
import (
	"encoding/json"
	"runtime"
	"unsafe"
)

// Tokenizer represents a text tokenizer
type Tokenizer struct {
	handle unsafe.Pointer
	tf     *TrustformeRS
}

// TokenizerInfo contains tokenizer information
type TokenizerInfo struct {
	Type       string            `json:"type"`
	VocabSize  int               `json:"vocab_size"`
	ModelName  string            `json:"model_name"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// SpecialTokens contains special token information
type SpecialTokens struct {
	BOS      *int `json:"bos,omitempty"`       // Beginning of sequence
	EOS      *int `json:"eos,omitempty"`       // End of sequence
	UNK      *int `json:"unk,omitempty"`       // Unknown token
	SEP      *int `json:"sep,omitempty"`       // Separator
	PAD      *int `json:"pad,omitempty"`       // Padding
	CLS      *int `json:"cls,omitempty"`       // Classification
	MASK     *int `json:"mask,omitempty"`      // Mask token
	Additional map[string]int `json:"additional,omitempty"` // Additional special tokens
}

// LoadTokenizerFromHub loads a tokenizer from Hugging Face Hub
func (tf *TrustformeRS) LoadTokenizerFromHub(modelName string) (*Tokenizer, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))

	var cError C.TrustformersError
	handle := C.trustformers_load_tokenizer_from_hub(cModelName, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	tokenizer := &Tokenizer{
		handle: handle,
		tf:     tf,
	}

	runtime.SetFinalizer(tokenizer, (*Tokenizer).finalize)
	return tokenizer, nil
}

// LoadTokenizerFromPath loads a tokenizer from a local path
func (tf *TrustformeRS) LoadTokenizerFromPath(tokenizerPath string) (*Tokenizer, error) {
	if !tf.initialized {
		return nil, errors.New("TrustformeRS not initialized")
	}

	cTokenizerPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cTokenizerPath))

	var cError C.TrustformersError
	handle := C.trustformers_load_tokenizer_from_path(cTokenizerPath, &cError)
	if err := checkError(cError); err != nil {
		return nil, err
	}

	if handle == nil {
		return nil, ErrRuntimeError
	}

	tokenizer := &Tokenizer{
		handle: handle,
		tf:     tf,
	}

	runtime.SetFinalizer(tokenizer, (*Tokenizer).finalize)
	return tokenizer, nil
}

// Free releases the tokenizer resources
func (t *Tokenizer) Free() error {
	if t.handle == nil {
		return nil
	}

	err := C.trustformers_tokenizer_free(t.handle)
	if err := checkError(err); err != nil {
		return err
	}

	t.handle = nil
	runtime.SetFinalizer(t, nil)
	return nil
}

// finalize is called by the finalizer
func (t *Tokenizer) finalize() {
	if t.handle != nil {
		t.Free()
	}
}

// Encode converts text to tokens
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.handle == nil {
		return nil, errors.New("tokenizer not loaded")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var cTokens *C.int
	var cNumTokens C.int

	err := C.trustformers_tokenizer_encode(t.handle, cText, &cTokens, &cNumTokens)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer C.trustformers_free_tokens(cTokens)

	if cTokens == nil || cNumTokens == 0 {
		return []int{}, nil
	}

	// Convert C array to Go slice
	tokens := make([]int, int(cNumTokens))
	cTokensSlice := (*[1 << 30]C.int)(unsafe.Pointer(cTokens))[:cNumTokens:cNumTokens]
	for i, token := range cTokensSlice {
		tokens[i] = int(token)
	}

	return tokens, nil
}

// Decode converts tokens to text
func (t *Tokenizer) Decode(tokens []int) (string, error) {
	if t.handle == nil {
		return "", errors.New("tokenizer not loaded")
	}

	if len(tokens) == 0 {
		return "", nil
	}

	// Convert Go slice to C array
	cTokens := make([]C.int, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}

	var cText *C.char
	err := C.trustformers_tokenizer_decode(t.handle, &cTokens[0], C.int(len(tokens)), &cText)
	if err := checkError(err); err != nil {
		return "", err
	}
	defer freeCString(cText)

	if cText == nil {
		return "", nil
	}

	return C.GoString(cText), nil
}

// EncodeBatch encodes multiple texts to tokens
func (t *Tokenizer) EncodeBatch(texts []string) ([][]int, error) {
	if t.handle == nil {
		return nil, errors.New("tokenizer not loaded")
	}

	if len(texts) == 0 {
		return [][]int{}, nil
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

	var cTokenBatches **C.int
	var cNumTokens *C.int

	err := C.trustformers_tokenizer_encode_batch(t.handle, &cTexts[0], C.int(len(texts)), &cTokenBatches, &cNumTokens)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer C.trustformers_free_token_batches(cTokenBatches, C.int(len(texts)))

	// Convert C arrays to Go slices
	result := make([][]int, len(texts))
	cTokenBatchesSlice := (*[1 << 30]*C.int)(unsafe.Pointer(cTokenBatches))[:len(texts):len(texts)]
	cNumTokensSlice := (*[1 << 30]C.int)(unsafe.Pointer(cNumTokens))[:len(texts):len(texts)]

	for i := 0; i < len(texts); i++ {
		numTokens := int(cNumTokensSlice[i])
		if numTokens > 0 && cTokenBatchesSlice[i] != nil {
			tokens := make([]int, numTokens)
			cTokensSlice := (*[1 << 30]C.int)(unsafe.Pointer(cTokenBatchesSlice[i]))[:numTokens:numTokens]
			for j, token := range cTokensSlice {
				tokens[j] = int(token)
			}
			result[i] = tokens
		} else {
			result[i] = []int{}
		}
	}

	return result, nil
}

// DecodeBatch decodes multiple token sequences to texts
func (t *Tokenizer) DecodeBatch(tokenBatches [][]int) ([]string, error) {
	if t.handle == nil {
		return nil, errors.New("tokenizer not loaded")
	}

	if len(tokenBatches) == 0 {
		return []string{}, nil
	}

	// Convert Go token batches to C arrays
	cTokenBatches := make([]*C.int, len(tokenBatches))
	cNumTokens := make([]C.int, len(tokenBatches))

	for i, tokens := range tokenBatches {
		if len(tokens) > 0 {
			cTokens := make([]C.int, len(tokens))
			for j, token := range tokens {
				cTokens[j] = C.int(token)
			}
			cTokenBatches[i] = &cTokens[0]
			cNumTokens[i] = C.int(len(tokens))
		} else {
			cTokenBatches[i] = nil
			cNumTokens[i] = 0
		}
	}

	var cTexts **C.char
	err := C.trustformers_tokenizer_decode_batch(t.handle, &cTokenBatches[0], &cNumTokens[0], C.int(len(tokenBatches)), &cTexts)
	if err := checkError(err); err != nil {
		return nil, err
	}
	defer C.trustformers_free_text_batch(cTexts, C.int(len(tokenBatches)))

	// Convert C string array to Go slice
	result := make([]string, len(tokenBatches))
	cTextsSlice := (*[1 << 30]*C.char)(unsafe.Pointer(cTexts))[:len(tokenBatches):len(tokenBatches)]

	for i, cText := range cTextsSlice {
		if cText != nil {
			result[i] = C.GoString(cText)
		} else {
			result[i] = ""
		}
	}

	return result, nil
}

// GetVocabSize returns the vocabulary size
func (t *Tokenizer) GetVocabSize() (int, error) {
	if t.handle == nil {
		return 0, errors.New("tokenizer not loaded")
	}

	var cVocabSize C.int
	err := C.trustformers_tokenizer_get_vocab_size(t.handle, &cVocabSize)
	if err := checkError(err); err != nil {
		return 0, err
	}

	return int(cVocabSize), nil
}

// GetSpecialTokens returns special token information
func (t *Tokenizer) GetSpecialTokens() (SpecialTokens, error) {
	if t.handle == nil {
		return SpecialTokens{}, errors.New("tokenizer not loaded")
	}

	var cSpecialTokensJSON *C.char
	err := C.trustformers_tokenizer_get_special_tokens(t.handle, &cSpecialTokensJSON)
	if err := checkError(err); err != nil {
		return SpecialTokens{}, err
	}
	defer freeCString(cSpecialTokensJSON)

	if cSpecialTokensJSON == nil {
		return SpecialTokens{}, nil
	}

	specialTokensJSON := C.GoString(cSpecialTokensJSON)
	var specialTokens SpecialTokens
	if err := json.Unmarshal([]byte(specialTokensJSON), &specialTokens); err != nil {
		return SpecialTokens{}, err
	}

	return specialTokens, nil
}

// AddSpecialToken adds a special token to the tokenizer
func (t *Tokenizer) AddSpecialToken(token string, tokenID int) error {
	if t.handle == nil {
		return errors.New("tokenizer not loaded")
	}

	cToken := C.CString(token)
	defer C.free(unsafe.Pointer(cToken))

	err := C.trustformers_tokenizer_add_special_token(t.handle, cToken, C.int(tokenID))
	return checkError(err)
}

// GetInfo returns tokenizer information
func (t *Tokenizer) GetInfo() (TokenizerInfo, error) {
	if t.handle == nil {
		return TokenizerInfo{}, errors.New("tokenizer not loaded")
	}

	var cInfoJSON *C.char
	err := C.trustformers_tokenizer_get_info(t.handle, &cInfoJSON)
	if err := checkError(err); err != nil {
		return TokenizerInfo{}, err
	}
	defer freeCString(cInfoJSON)

	if cInfoJSON == nil {
		return TokenizerInfo{}, errors.New("failed to get tokenizer info")
	}

	infoJSON := C.GoString(cInfoJSON)
	var info TokenizerInfo
	if err := json.Unmarshal([]byte(infoJSON), &info); err != nil {
		return TokenizerInfo{}, err
	}

	return info, nil
}

// IsLoaded checks if the tokenizer is loaded
func (t *Tokenizer) IsLoaded() bool {
	return t.handle != nil
}