// 파일 업로드 관련 함수
function handleFileUpload(formId, endpoint, onSuccess, onError) {
    const form = document.getElementById(formId);
    const loadingSpinner = document.getElementById('loading-spinner');
    const progressMessage = document.querySelector('#loading-spinner p');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        if (!formData.get('file')) {
            alert('파일을 선택해주세요.');
            return;
        }

        loadingSpinner.style.display = 'block';
        progressMessage.textContent = "파일을 업로드하는 중...";

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('서버 오류가 발생했습니다.');
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            if (onSuccess) {
                onSuccess(data);
            }
        } catch (error) {
            if (onError) {
                onError(error);
            } else {
                alert('오류가 발생했습니다: ' + error.message);
            }
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });
}

// 상태 확인 함수
async function checkStatus(taskId, onSuccess, onError) {
    try {
        const response = await fetch(`/status/${taskId}`);
        const data = await response.json();
        
        if (data.state === 'SUCCESS') {
            if (onSuccess) {
                onSuccess(data.result);
            }
            return true;
        } else if (data.state === 'FAILURE') {
            if (onError) {
                onError(new Error('분석 중 오류가 발생했습니다.'));
            }
            return true;
        }
        
        return false;
    } catch (error) {
        if (onError) {
            onError(error);
        }
        return true;
    }
}

// 주기적 상태 확인 함수
function pollStatus(taskId, interval, onSuccess, onError) {
    const check = async () => {
        const shouldStop = await checkStatus(taskId, onSuccess, onError);
        if (!shouldStop) {
            setTimeout(check, interval);
        }
    };
    
    check();
}

// 결과 표시 함수
function displayResults(containerId, results) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // 결과가 문자열인 경우
    if (typeof results === 'string') {
        container.innerHTML = results;
        return;
    }

    // 결과가 객체인 경우
    let html = '';
    for (const [key, value] of Object.entries(results)) {
        html += `
            <div class="result-section">
                <h3>${key}</h3>
                <div class="result-content">
                    ${typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// 에러 표시 함수
function displayError(containerId, error) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="error-message">
            <h3>오류가 발생했습니다</h3>
            <p>${error.message}</p>
        </div>
    `;
} 