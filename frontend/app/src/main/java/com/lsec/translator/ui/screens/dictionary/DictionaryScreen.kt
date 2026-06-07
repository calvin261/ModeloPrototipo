package com.lsec.translator.ui.screens.dictionary

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.lsec.translator.data.local.entity.SignCategory
import com.lsec.translator.data.local.entity.SignEntity

/**
 * Pantalla del diccionario LSEC.
 * Permite explorar, buscar y filtrar señas.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DictionaryScreen(
    onNavigateBack: () -> Unit,
    viewModel: DictionaryViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var searchQuery by remember { mutableStateOf("") }
    var showCategoryFilter by remember { mutableStateOf(false) }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Diccionario LSEC") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Volver"
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { showCategoryFilter = !showCategoryFilter }) {
                        Icon(
                            imageVector = Icons.Default.FilterList,
                            contentDescription = "Filtrar por categoría"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Barra de búsqueda
            SearchBar(
                query = searchQuery,
                onQueryChange = { 
                    searchQuery = it
                    viewModel.search(it)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            )
            
            // Filtros de categoría
            if (showCategoryFilter) {
                CategoryFilterRow(
                    selectedCategory = uiState.selectedCategory,
                    onCategorySelected = { viewModel.filterByCategory(it) },
                    modifier = Modifier.padding(horizontal = 16.dp)
                )
                
                Spacer(modifier = Modifier.height(8.dp))
            }
            
            // Tabs: Todas / Favoritas / Más usadas
            TabRow(
                selectedTabIndex = uiState.selectedTab
            ) {
                Tab(
                    selected = uiState.selectedTab == 0,
                    onClick = { viewModel.selectTab(0) },
                    text = { Text("Todas") },
                    icon = { Icon(Icons.Default.List, null) }
                )
                Tab(
                    selected = uiState.selectedTab == 1,
                    onClick = { viewModel.selectTab(1) },
                    text = { Text("Favoritas") },
                    icon = { Icon(Icons.Default.Favorite, null) }
                )
                Tab(
                    selected = uiState.selectedTab == 2,
                    onClick = { viewModel.selectTab(2) },
                    text = { Text("Frecuentes") },
                    icon = { Icon(Icons.Default.TrendingUp, null) }
                )
            }
            
            // Lista de señas
            if (uiState.isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator()
                }
            } else if (uiState.signs.isEmpty()) {
                EmptyState(
                    message = if (searchQuery.isNotEmpty()) 
                        "No se encontraron señas para \"$searchQuery\"" 
                    else 
                        "No hay señas en esta categoría"
                )
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(uiState.signs, key = { it.id }) { sign ->
                        SignCard(
                            sign = sign,
                            onToggleFavorite = { viewModel.toggleFavorite(sign.id) },
                            onSpeak = { viewModel.speakSign(sign) }
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SearchBar(
    query: String,
    onQueryChange: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    OutlinedTextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = modifier,
        placeholder = { Text("Buscar seña, palabra o frase...") },
        leadingIcon = {
            Icon(Icons.Default.Search, contentDescription = "Buscar")
        },
        trailingIcon = {
            if (query.isNotEmpty()) {
                IconButton(onClick = { onQueryChange("") }) {
                    Icon(Icons.Default.Clear, contentDescription = "Limpiar")
                }
            }
        },
        singleLine = true,
        shape = RoundedCornerShape(24.dp)
    )
}

@Composable
private fun CategoryFilterRow(
    selectedCategory: SignCategory?,
    onCategorySelected: (SignCategory?) -> Unit,
    modifier: Modifier = Modifier
) {
    val categories = SignCategory.values()
    
    LazyColumn(
        modifier = modifier.height(200.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        item {
            FilterChip(
                selected = selectedCategory == null,
                onClick = { onCategorySelected(null) },
                label = { Text("Todas") },
                leadingIcon = if (selectedCategory == null) {
                    { Icon(Icons.Default.Check, null, Modifier.size(16.dp)) }
                } else null
            )
        }
        items(categories.size) { index ->
            val category = categories[index]
            FilterChip(
                selected = selectedCategory == category,
                onClick = { onCategorySelected(category) },
                label = { Text(getCategoryDisplayName(category)) },
                leadingIcon = if (selectedCategory == category) {
                    { Icon(Icons.Default.Check, null, Modifier.size(16.dp)) }
                } else null
            )
        }
    }
}

@Composable
private fun SignCard(
    sign: SignEntity,
    onToggleFavorite: () -> Unit,
    onSpeak: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = sign.name,
                        style = MaterialTheme.typography.titleMedium
                    )
                    Text(
                        text = sign.spanishTranslation,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    if (sign.ecuadorianVariant.isNotEmpty() && 
                        sign.ecuadorianVariant != sign.spanishTranslation) {
                        Text(
                            text = "\"${sign.ecuadorianVariant}\"",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
                
                Row {
                    IconButton(onClick = onSpeak) {
                        Icon(
                            imageVector = Icons.Default.VolumeUp,
                            contentDescription = "Escuchar",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    IconButton(onClick = onToggleFavorite) {
                        Icon(
                            imageVector = if (sign.isFavorite) 
                                Icons.Default.Favorite 
                            else 
                                Icons.Default.FavoriteBorder,
                            contentDescription = "Favorito",
                            tint = if (sign.isFavorite) 
                                MaterialTheme.colorScheme.error 
                            else 
                                MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                AssistChip(
                    onClick = {},
                    label = { Text(getCategoryDisplayName(sign.category)) },
                    leadingIcon = {
                        Icon(
                            getCategoryIcon(sign.category),
                            null,
                            Modifier.size(16.dp)
                        )
                    }
                )
                
                if (sign.isDynamic) {
                    AssistChip(
                        onClick = {},
                        label = { Text("Dinámica") },
                        leadingIcon = {
                            Icon(
                                Icons.Default.Moving,
                                null,
                                Modifier.size(16.dp)
                            )
                        }
                    )
                }
            }
            
            if (sign.description.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = sign.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun EmptyState(message: String) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Default.SearchOff,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = message,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

private fun getCategoryDisplayName(category: SignCategory): String {
    return when (category) {
        SignCategory.ALPHABET -> "Abecedario"
        SignCategory.NUMBERS -> "Números"
        SignCategory.GREETINGS -> "Saludos"
        SignCategory.FAMILY -> "Familia"
        SignCategory.FOOD -> "Comida"
        SignCategory.HEALTH -> "Salud"
        SignCategory.EDUCATION -> "Educación"
        SignCategory.EMOTIONS -> "Emociones"
        SignCategory.TIME -> "Tiempo"
        SignCategory.PLACES -> "Lugares"
        SignCategory.ACTIONS -> "Acciones"
        SignCategory.QUESTIONS -> "Preguntas"
        SignCategory.COMMON_PHRASES -> "Frases comunes"
        SignCategory.EMERGENCY -> "Emergencias"
        SignCategory.OTHER -> "Otros"
    }
}

private fun getCategoryIcon(category: SignCategory): androidx.compose.ui.graphics.vector.ImageVector {
    return when (category) {
        SignCategory.ALPHABET -> Icons.Default.Abc
        SignCategory.NUMBERS -> Icons.Default.Numbers
        SignCategory.GREETINGS -> Icons.Default.WavingHand
        SignCategory.FAMILY -> Icons.Default.FamilyRestroom
        SignCategory.FOOD -> Icons.Default.Restaurant
        SignCategory.HEALTH -> Icons.Default.LocalHospital
        SignCategory.EDUCATION -> Icons.Default.School
        SignCategory.EMOTIONS -> Icons.Default.EmojiEmotions
        SignCategory.TIME -> Icons.Default.Schedule
        SignCategory.PLACES -> Icons.Default.Place
        SignCategory.ACTIONS -> Icons.Default.DirectionsRun
        SignCategory.QUESTIONS -> Icons.Default.QuestionMark
        SignCategory.COMMON_PHRASES -> Icons.Default.Chat
        SignCategory.EMERGENCY -> Icons.Default.Warning
        SignCategory.OTHER -> Icons.Default.MoreHoriz
    }
}
