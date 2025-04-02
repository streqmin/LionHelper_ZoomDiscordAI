document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('chat-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    const progressMessage = document.querySelector('#loading-spinner p');
    const resultsContainer = document.getElementById('chatAnalysis');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        if (!formData.get('chat_file')) {
            alert('채팅 기록 파일을 선택해주세요.');
            return;
        }

        loadingSpinner.style.display = 'block';
        resultsContainer.style.display = 'none';
        progressMessage.textContent = "채팅 파일을 읽는 중...";

        try {
            const response = await fetch('/analyze_chat', {
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

            progressMessage.textContent = "채팅 내용을 분석하는 중...";
            pollStatus(data.task_id, 1000, 
                (result) => {
                    progressMessage.textContent = "분석 결과를 처리하는 중...";
                    resultsContainer.innerHTML = formatChatAnalysis(result);
                    resultsContainer.style.display = 'block';
                    
                    progressMessage.textContent = "분석이 완료되었습니다!";
                    setTimeout(() => {
                        loadingSpinner.style.display = 'none';
                    }, 1000);
                },
                (error) => {
                    progressMessage.textContent = "오류가 발생했습니다";
                    alert('오류가 발생했습니다: ' + error.message);
                    loadingSpinner.style.display = 'none';
                }
            );
        } catch (error) {
            progressMessage.textContent = "오류가 발생했습니다";
            alert('오류가 발생했습니다: ' + error.message);
            loadingSpinner.style.display = 'none';
        }
    });
});

function formatChatAnalysis(content) {
    const sections = content.split(/(?=\d\. )/);
    let formattedContent = '';

    sections.forEach(section => {
        if (section.trim()) {
            const mainNumber = section.match(/^(\d+)\./)?.[1] || '';
            const mainTitle = section.match(/^\d+\. (.+)$/m)?.[1] || '';
            const sectionContent = section.replace(/^\d+\. .+\n/m, '').trim();
            
            if (!mainTitle) return;
            
            const subsections = sectionContent.split(/(?=\d+\.\d+ )/);
            
            formattedContent += `
                <div class="summary-section">
                    <h2><span class="section-number">${mainNumber}.</span>${mainTitle}</h2>
                    ${subsections.map(subsection => {
                        if (!subsection.trim()) return '';
                        
                        const subNumber = subsection.match(/^(\d+\.\d+) /)?.[1] || '';
                        const subTitle = subsection.match(/^\d+\.\d+ (.+)$/m)?.[1] || '';
                        const subContent = subsection.replace(/^\d+\.\d+ .+\n/m, '').trim();
                        
                        if (!subTitle || !subContent.trim()) return '';
                        
                        return `
                            <div class="subsection">
                                <div class="subsection-title">
                                    <span class="section-number">${subNumber}</span>${subTitle}
                                </div>
                                <ul>
                                    ${subContent.split('\n')
                                        .map(line => line.trim())
                                        .filter(line => line.startsWith('•') || line.startsWith('-'))
                                        .map(line => `<li>${line.replace(/^[•-]\s*/, '')}</li>`)
                                        .join('')}
                                </ul>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        }
    });

    return formattedContent;
} 